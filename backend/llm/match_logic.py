from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Union

from config import DB_PATH
from backend.llm.llm_client import evaluate_with_llm, evaluate_batch_with_llm, _evaluate_rule_based

PathArg = Union[str, Path]

BATCH_SIZE = 80   # Massive batches (80 patients/call) drastically minimize API calls


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_keywords(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    if isinstance(value, list):
        return [str(s).strip() for s in value if s is not None and str(s).strip()]
    return []


def fetch_all_patients(db_path: Optional[PathArg] = None) -> List[Dict]:
    db_path = db_path or DB_PATH
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT patient_id, nct_number, age, diagnosis, medications, notes FROM patients"
        )
        rows = cur.fetchall()

    patients: List[Dict] = []
    for patient_id, nct_number, age, diagnosis, medications, notes in rows:
        patients.append({
            "patient_id":  patient_id,
            "nct_number":  nct_number or "",
            "age":         age if age is not None else 0,
            "diagnosis":   diagnosis or "",
            "medications": medications or "",
            "notes":       notes or "",
        })
    return patients


# ── Fast pre-filter: reject obvious non-matches WITHOUT calling the LLM ────────

def _fast_evaluate(patient: Dict, trial: Dict):
    """
    Returns a result dict immediately — WITHOUT an LLM call —
    if the patient is clearly eligible or ineligible based on hard rules.
    Returns None if the patient needs LLM semantic evaluation (medical terminology).
    """
    try:
        age     = int(patient.get("age") or 0)
        age_min = int(trial.get("age_min", 0) or 0)
        age_max = int(trial.get("age_max", 999) or 999)
    except Exception:
        age, age_min, age_max = 0, 0, 999

    # 1. Age out of range → instant rejection
    if not (age_min <= age <= age_max):
        return _build_fast_result(
            patient, age,
            reason=f"Age {age} is outside trial range {age_min}-{age_max}.",
            decision="Not eligible",
            age_ok=False,
        )

    full_text = " ".join([
        patient.get("diagnosis",   ""),
        patient.get("medications", ""),
        patient.get("notes",       ""),
    ]).lower()

    # 2. Exact exclusion keyword hit → instant rejection
    exc_kws = [k.lower() for k in trial.get("exclusion_keywords", []) if k]
    hit_excl = [kw for kw in exc_kws if kw in full_text]
    if hit_excl:
        return _build_fast_result(
            patient, age,
            reason=f"Excluded: patient record contains '{hit_excl[0]}'.",
            decision="Not eligible",
            age_ok=True,
            matched_excl=hit_excl,
        )

    # 3. No text at all → can't match anything
    if not full_text.strip():
        return _build_fast_result(
            patient, age,
            reason="Patient record has no clinical data to match against trial criteria.",
            decision="Not eligible",
            age_ok=True,
        )

    # 4. Exact inclusion keyword hit → Fast Accept
    inc_kws = [k.lower() for k in trial.get("inclusion_keywords", []) if k]
    hit_incl = [kw for kw in inc_kws if kw in full_text]
    if hit_incl:
        return _build_fast_result(
            patient, age,
            reason=f"Rule-based match for keyword(s): {', '.join(hit_incl[:3])}",
            decision="Eligible",
            age_ok=True,
            matched_incl=hit_incl,
        )

    return None   # needs LLM


def _build_fast_result(patient, age, *, reason, decision, age_ok, matched_excl=None, matched_incl=None):
    return {
        "patient_id":        patient.get("patient_id"),
        "nct_number":        patient.get("nct_number", ""),
        "age":               age,
        "diagnosis":         patient.get("diagnosis", ""),
        "medications":       patient.get("medications", ""),
        "notes":             patient.get("notes", ""),
        "score":             1.0 if decision == "Eligible" else 0.0,
        "decision":          decision,
        "reason":            reason,
        "matched_inclusion": matched_incl or [],
        "matched_exclusion": matched_excl or [],
        "age_ok":            age_ok,
        "inclusion_score":   1.0 if decision == "Eligible" else 0.0,
        "exclusion_score":   1.0 if matched_excl else 0.0,
        "_source":           "rule_based" if decision == "Eligible" else "pre_filter",
    }


def _llm_result_to_row(patient: Dict, trial: Dict, llm: Dict, min_score: float) -> Dict:
    try:
        age     = int(patient.get("age") or 0)
        age_min = int(trial.get("age_min", 0) or 0)
        age_max = int(trial.get("age_max", 999) or 999)
        age_ok  = age_min <= age <= age_max
    except Exception:
        age, age_ok = 0, True

    eligible     = llm.get("eligible", False)
    score        = float(llm.get("score", 0.0))
    reason       = llm.get("reason", "")
    matched_incl = llm.get("matched_inclusion", [])
    matched_excl = llm.get("matched_exclusion", [])

    if not age_ok:
        eligible = False
        reason   = f"Age {age} is outside trial range {age_min}-{age_max}. " + reason
    if score < min_score:
        eligible = False

    return {
        "patient_id":        patient.get("patient_id"),
        "nct_number":        patient.get("nct_number", ""),
        "age":               age,
        "diagnosis":         patient.get("diagnosis", ""),
        "medications":       patient.get("medications", ""),
        "notes":             patient.get("notes", ""),
        "score":             round(score, 3),
        "decision":          "Eligible" if eligible else "Not eligible",
        "reason":            reason,
        "matched_inclusion": matched_incl,
        "matched_exclusion": matched_excl,
        "age_ok":            age_ok,
        "inclusion_score":   score,
        "exclusion_score":   1.0 if matched_excl else 0.0,
        "_source":           "llm",   # LLM semantic evaluation
    }


# ── Single-patient evaluator (used by /chat endpoint) ─────────────────────────

def evaluate_patient(patient: Dict, trial: Dict, *, min_inclusion_score: float = 0.0) -> Dict:
    fast = _fast_evaluate(patient, trial)
    if fast:
        return fast
    llm = evaluate_with_llm(patient, trial)
    return _llm_result_to_row(patient, trial, llm, min_inclusion_score)


# ── Batch runner — FAST ────────────────────────────────────────────────────────

def run_match_all(
    trial: Dict,
    db_path: Optional[PathArg] = None,
    *,
    eligible_only:       bool  = False,
    min_inclusion_score: float = 0.0,
) -> List[Dict]:
    """
    Evaluate all patients with two speed layers:
      1. Fast pre-filter  — age / exclusion check (no API call)
      2. Batch LLM calls  — 10 patients per API call (10x fewer round-trips)
    """
    provider = os.environ.get("LLM_PROVIDER", "rule_based")
    patients = fetch_all_patients(db_path)
    total    = len(patients)
    print(f"[match] {total} patients | trial: {trial.get('id')} | LLM: {provider}")

    if not patients:
        print("[match] No patients found in database!")
        return []

    results: List[Dict] = []

    # ── Rule-based: no batching needed, already instant ──────────────────────
    if provider == "rule_based":
        for p in patients:
            results.append(evaluate_patient(p, trial, min_inclusion_score=min_inclusion_score))
        if eligible_only:
            results = [r for r in results if r["decision"] == "Eligible"]
        return sorted(results, key=lambda x: x["score"], reverse=True)

    # ── LLM path ─────────────────────────────────────────────────────────────
    # Step 1: Fast pre-filter (no API call)
    needs_llm:   List[Dict] = []
    fast_results: List[Dict] = []

    for p in patients:
        fast = _fast_evaluate(p, trial)
        if fast:
            fast_results.append(fast)
        else:
            needs_llm.append(p)

    skipped = len(fast_results)
    llm_count = len(needs_llm)
    batches = (llm_count + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[match] rule-based: {skipped} instantly evaluated | "
          f"{llm_count} need LLM → {batches} batch calls (batch_size={BATCH_SIZE})")

    # Step 2: Batch LLM calls in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Split needs_llm into chunks of BATCH_SIZE
    chunks = [needs_llm[i:i + BATCH_SIZE] for i in range(0, llm_count, BATCH_SIZE)]
    llm_rows: List[Dict] = []
    done_batches = 0

    def _eval_batch(chunk):
        llm_results = evaluate_batch_with_llm(chunk, trial)
        return [
            _llm_result_to_row(p, trial, lr, min_inclusion_score)
            for p, lr in zip(chunk, llm_results)
        ]

    # Groq free tier = 6 000 TPM shared across ALL threads.
    # The rate limiter in llm_client.py handles pacing, but parallel threads
    # still race each other and all block simultaneously → no speed gain.
    # Sequential (1 worker) lets the limiter meter calls smoothly with zero 429s.
    default_workers = "1" if provider == "groq" else "10"
    max_workers = min(int(os.environ.get("LLM_WORKERS", default_workers)), max(1, batches))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_eval_batch, chunk) for chunk in chunks]
        for future in as_completed(futures):
            llm_rows.extend(future.result())
            done_batches += 1
            if done_batches % 5 == 0 or done_batches == batches:
                processed = done_batches * BATCH_SIZE
                print(f"[match] batch progress: ~{min(processed, llm_count)}/{llm_count} patients")

    results = fast_results + llm_rows

    if eligible_only:
        results = [r for r in results if r["decision"] == "Eligible"]

    return sorted(results, key=lambda x: x["score"], reverse=True)


# ── Utility ───────────────────────────────────────────────────────────────────

def list_available_trials(trials_dir: Optional[PathArg] = None) -> List[str]:
    from config import TRIALS_DIR
    trials_dir = Path(trials_dir or TRIALS_DIR)
    if not trials_dir.exists():
        return []
    return [f.stem for f in trials_dir.glob("*.json")]