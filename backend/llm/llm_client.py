from __future__ import annotations

import json
import re
import os
import shelve
import hashlib
import time
import threading
from pathlib import Path
from typing import Dict, List

# ── Persistent disk cache — survives server restarts ─────────────────────────
_CACHE_DIR = Path(__file__).parent.parent / "data" / "results"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_PATH = str(_CACHE_DIR / "llm_cache")

# In-memory layer on top of disk cache for ultra-fast repeat hits
_mem_cache: Dict[str, Dict] = {}


class _GroqRateLimiter:
    """
    Thread-safe sliding-window token and request tracker.
    Call .acquire(n_tokens) before each Groq API call.
    Sleeps only as long as needed to stay under the TPM and RPM caps.
    """
    def __init__(self, tpm_limit: int = 100000, rpm_limit: int = 28, window_sec: float = 60.0):
        # Groq llama-3.1-8b-instant free tier: 131,072 TPM, 30 RPM — use buffers
        self._tpm_limit = tpm_limit
        self._rpm_limit = rpm_limit
        self._window    = window_sec
        self._lock      = threading.Lock()
        self._log: list = []   # list of (timestamp, tokens) tuples

    def _trim(self, now: float):
        cutoff = now - self._window
        self._log = [(t, n) for t, n in self._log if t > cutoff]

    def acquire(self, n_tokens: int):
        while True:
            with self._lock:
                now  = time.monotonic()
                self._trim(now)
                used_tokens = sum(n for _, n in self._log)
                used_reqs   = len(self._log)
                
                if used_tokens + n_tokens <= self._tpm_limit and used_reqs + 1 <= self._rpm_limit:
                    self._log.append((now, n_tokens))
                    return   # budget available — proceed immediately
                
                # Oldest entry that needs to expire
                if used_reqs + 1 > self._rpm_limit:
                    sleep_for = (self._log[0][0] + self._window) - now + 0.1
                    reason = f"RPM limit ({used_reqs}>={self._rpm_limit})"
                else:
                    sleep_for = (self._log[0][0] + self._window) - now + 0.1
                    reason = f"TPM limit ({used_tokens}+{n_tokens}>{self._tpm_limit})"
                    
            # Sleep outside the lock so other threads can check too
            print(f"[RateLimit] Budget full {reason} — waiting {sleep_for:.1f}s")
            time.sleep(max(sleep_for, 0.5))

_groq_limiter = _GroqRateLimiter(tpm_limit=100000, rpm_limit=28)


def _cache_key(patient: Dict, trial: Dict) -> str:
    raw = f"{patient.get('patient_id')}::{trial.get('id', trial.get('title', ''))}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_from_cache(key: str):
    if key in _mem_cache:
        return _mem_cache[key]
    try:
        with shelve.open(_CACHE_PATH) as db:
            if key in db:
                result = db[key]
                _mem_cache[key] = result
                return result
    except Exception:
        pass
    return None


def _save_to_cache(key: str, result: Dict):
    _mem_cache[key] = result
    try:
        with shelve.open(_CACHE_PATH) as db:
            db[key] = result
    except Exception:
        pass


# ── JSON extractor ─────────────────────────────────────────────────────────────
def _parse_json_response(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


# ── BATCH prompt builder — compact to minimise token usage ────────────────────
def _build_batch_prompt(patients: List[Dict], trial: Dict) -> str:
    inc = ",".join(trial.get("inclusion_keywords", [])[:6])
    exc = ",".join(trial.get("exclusion_keywords", [])[:4])
    trial_info = (
        f"T:{trial.get('id')} age:{trial.get('age_min',0)}-{trial.get('age_max',100)} "
        f"inc:[{inc}] exc:[{exc}]"
    )
    patient_lines = []
    for p in patients:
        patient_lines.append(
            f"{p.get('patient_id')} a:{p.get('age')} "
            f"dx:{str(p.get('diagnosis',''))[:60]} "
            f"med:{str(p.get('medications',''))[:40]}"
        )

    return (
        f"Eligibility check. Synonyms ok (COPD=pulmonary, HTN=hypertension).\n"
        f"{trial_info}\n"
        f"Patients:\n"
        + "\n".join(f"{i+1}.{line}" for i, line in enumerate(patient_lines))
        + '\nJSON only: {"results":[{"patient_id":"..","eligible":bool,"score":0.0-1.0,"reason":"short","matched_inclusion":[],"matched_exclusion":[]}]}'
    )


# ── Single-patient prompt — compact ──────────────────────────────────────────
def _build_prompt(patient: Dict, trial: Dict) -> str:
    inc = ",".join(trial.get("inclusion_keywords", [])[:6])
    exc = ",".join(trial.get("exclusion_keywords", [])[:4])
    return (
        f"Eligibility check. Synonyms ok.\n"
        f"P: age={patient.get('age')} dx={str(patient.get('diagnosis',''))[:80]} "
        f"med={str(patient.get('medications',''))[:50]}\n"
        f"T:{trial.get('id')} age:{trial.get('age_min',0)}-{trial.get('age_max',100)} inc:[{inc}] exc:[{exc}]\n"
        f'JSON: {{"eligible":bool,"score":0.0-1.0,"reason":"1 sentence","matched_inclusion":[],"matched_exclusion":[]}}'
    )


# ── Rate-limit retry helper ────────────────────────────────────────────────────
def _parse_retry_after(err_str: str) -> float:
    """Extract the wait time in seconds from a Groq 429 error message."""
    m = re.search(r"Please try again in ([\d.]+)s", str(err_str))
    if m:
        return min(float(m.group(1)) + 0.5, 30.0)   # add 0.5s buffer, cap at 30s
    return 5.0   # safe default


# ── Groq — BATCH (patients per call → fewer API calls) ────────────────────────
def _evaluate_groq_batch(patients: List[Dict], trial: Dict, model: str) -> List[Dict]:
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)
    prompt = _build_batch_prompt(patients, trial)

    # Estimate tokens: ~1 token per 4 chars for prompt + ~60 tokens output per patient
    est_input  = len(prompt) // 4
    est_output = 60 * len(patients)
    est_total  = est_input + est_output

    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        # Proactively wait if we'd exceed TPM budget
        _groq_limiter.acquire(est_total)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Clinical trial eligibility expert. JSON only."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=min(90 * len(patients), 7000),
                response_format={"type": "json_object"},
                timeout=45.0,
            )
            break   # success — exit retry loop
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < MAX_RETRIES - 1:
                wait = _parse_retry_after(err)
                print(f"[LLM] 429 despite limiter — waiting {wait:.1f}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise   # re-raise on non-429 or final attempt

    raw = response.choices[0].message.content
    parsed = _parse_json_response(raw)
    results = parsed.get("results", [])

    # Map back by index — fallback by patient_id match
    out: List[Dict] = []
    for i, p in enumerate(patients):
        match = None
        # Try by patient_id first
        for r in results:
            if str(r.get("patient_id", "")).upper() == str(p.get("patient_id", "")).upper():
                match = r
                break
        # Fall back to position
        if match is None and i < len(results):
            match = results[i]
        if match is None:
            match = {"eligible": False, "score": 0.0, "reason": "No LLM result", "matched_inclusion": [], "matched_exclusion": []}
        match.setdefault("eligible", False)
        match.setdefault("score", 0.0)
        match.setdefault("reason", "")
        match.setdefault("matched_inclusion", [])
        match.setdefault("matched_exclusion", [])
        out.append(match)
    return out


# ── Groq — single patient (used as fallback if batch fails) ───────────────────
def _evaluate_groq(patient: Dict, trial: Dict, model: str) -> Dict:
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    client  = Groq(api_key=api_key)
    prompt  = _build_prompt(patient, trial)
    est_tok = len(prompt) // 4 + 120   # input estimate + output buffer

    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        _groq_limiter.acquire(est_tok)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Clinical trial eligibility expert. JSON only."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=80,
                response_format={"type": "json_object"},
                timeout=20.0,
            )
            break
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < MAX_RETRIES - 1:
                wait = _parse_retry_after(err)
                print(f"[LLM] 429 (single) — waiting {wait:.1f}s (attempt {attempt+1})")
                time.sleep(wait)
            else:
                raise
    raw = response.choices[0].message.content
    return _parse_json_response(raw)


# ── OpenAI ─────────────────────────────────────────────────────────────────────
def _evaluate_openai(patient: Dict, trial: Dict, model: str = "gpt-4o") -> Dict:
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a clinical trial eligibility expert. Always respond with valid JSON only."},
            {"role": "user",   "content": _build_prompt(patient, trial)},
        ],
        temperature=0.0,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    return _parse_json_response(response.choices[0].message.content)


# ── Ollama ─────────────────────────────────────────────────────────────────────
def _evaluate_ollama(patient: Dict, trial: Dict, model: str = "llama3.2:3b") -> Dict:
    import requests
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    resp = requests.post(
        f"{ollama_url}/api/generate",
        json={"model": model, "prompt": _build_prompt(patient, trial), "stream": False},
        timeout=30,
    )
    resp.raise_for_status()
    return _parse_json_response(resp.json().get("response", ""))


# ── Rule-based fallback ────────────────────────────────────────────────────────
def _evaluate_rule_based(patient: Dict, trial: Dict) -> Dict:
    full_text = " ".join([
        patient.get("diagnosis", ""),
        patient.get("medications", ""),
        patient.get("notes", ""),
    ]).lower()

    inc_kws = [k.lower() for k in trial.get("inclusion_keywords", []) if k]
    exc_kws = [k.lower() for k in trial.get("exclusion_keywords", []) if k]
    matched_inc = [kw for kw in inc_kws if kw in full_text]
    matched_exc = [kw for kw in exc_kws if kw in full_text]

    try:
        age = int(patient.get("age") or 0)
        age_ok = int(trial.get("age_min", 0)) <= age <= int(trial.get("age_max", 999))
    except Exception:
        age_ok = True

    inc_score = len(matched_inc) / len(inc_kws) if inc_kws else 0.0
    eligible = age_ok and inc_score > 0.0 and len(matched_exc) == 0

    return {
        "eligible": eligible,
        "score": round(inc_score, 3),
        "reason": (f"Rule-based: age {'ok' if age_ok else 'out of range'}, "
                   f"{len(matched_inc)}/{len(inc_kws)} inclusion keywords matched, "
                   f"{len(matched_exc)} exclusion keywords hit."),
        "matched_inclusion": matched_inc,
        "matched_exclusion": matched_exc,
    }


# ── Public: evaluate ONE patient (with cache) ──────────────────────────────────
def evaluate_with_llm(patient: Dict, trial: Dict) -> Dict:
    key = _cache_key(patient, trial)
    cached = _load_from_cache(key)
    if cached:
        return cached

    provider = os.environ.get("LLM_PROVIDER", "rule_based").lower()
    try:
        if provider == "openai":
            result = _evaluate_openai(patient, trial, os.environ.get("OPENAI_MODEL", "gpt-4o"))
        elif provider == "groq":
            result = _evaluate_groq(patient, trial, os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"))
        elif provider == "ollama":
            result = _evaluate_ollama(patient, trial, os.environ.get("OLLAMA_MODEL", "llama3.2:3b"))
        else:
            result = _evaluate_rule_based(patient, trial)

        result.setdefault("eligible", False)
        result.setdefault("score", 0.0)
        result.setdefault("reason", "")
        result.setdefault("matched_inclusion", [])
        result.setdefault("matched_exclusion", [])
        _save_to_cache(key, result)
        return result

    except Exception as e:
        print(f"[LLM] Error ({provider}): {e} — falling back to rule-based")
        fallback = _evaluate_rule_based(patient, trial)
        fallback["reason"] = f"[Fallback] {fallback['reason']} (LLM error: {e})"
        return fallback


# ── Public: evaluate a BATCH of patients (Groq only, 10x fewer API calls) ─────
def evaluate_batch_with_llm(patients: List[Dict], trial: Dict) -> List[Dict]:
    """
    Evaluate up to 10 patients in a single LLM call.
    Results are cached individually so repeat calls are instant.
    """
    provider = os.environ.get("LLM_PROVIDER", "rule_based").lower()

    # Check cache for all — only send uncached patients to LLM
    results = [None] * len(patients)
    uncached_indices = []
    uncached_patients = []

    for i, p in enumerate(patients):
        key = _cache_key(p, trial)
        cached = _load_from_cache(key)
        if cached:
            results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_patients.append(p)

    if not uncached_patients:
        return results  # All from cache — instant!

    # Batch call for uncached patients
    if provider == "groq" and uncached_patients:
        model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        try:
            batch_results = _evaluate_groq_batch(uncached_patients, trial, model)
            for idx, (orig_i, p, r) in enumerate(zip(uncached_indices, uncached_patients, batch_results)):
                r.setdefault("eligible", False)
                r.setdefault("score", 0.0)
                r.setdefault("reason", "")
                r.setdefault("matched_inclusion", [])
                r.setdefault("matched_exclusion", [])
                _save_to_cache(_cache_key(p, trial), r)
                results[orig_i] = r
            return results
        except Exception as e:
            print(f"[LLM] Batch failed ({e}), falling back to individual calls")

    # Fallback: evaluate individually
    for orig_i, p in zip(uncached_indices, uncached_patients):
        results[orig_i] = evaluate_with_llm(p, trial)

    return results
