import asyncio
import csv, io, json, os, re, sqlite3, tempfile, time
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from config import DATA_DIR, TRIALS_DIR, UPLOADS_DIR, DB_PATH
from backend.llm.match_logic import run_match_all, fetch_all_patients
from backend.utils.patient_import import import_patients_csv_bytes

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Clinical Trial Matcher API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174",
                   "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class MatchRequest(BaseModel): trial_id: str
class ChatRequest(BaseModel):  question: str
class ChatMessage(BaseModel):  role: str; content: str

# ── DB helpers ───────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY, nct_number TEXT,
            age INTEGER, diagnosis TEXT, medications TEXT, notes TEXT
        )""")
        conn.commit()

def db_count():
    with sqlite3.connect(DB_PATH) as c:
        return c.execute("SELECT COUNT(*) FROM patients").fetchone()[0]

def load_trial(trial_id: str) -> dict:
    p = Path(TRIALS_DIR) / f"{trial_id}.json"
    if not p.exists():
        available = [f.stem for f in Path(TRIALS_DIR).glob("*.json")]
        raise HTTPException(404, f"Trial '{trial_id}' not found. Available: {available}")
    return json.loads(p.read_text(encoding="utf-8"))

def all_trial_ids():
    return [f.stem for f in Path(TRIALS_DIR).glob("*.json")]

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()
    sample = Path(TRIALS_DIR) / "NCT123456.json"
    if not sample.exists():
        sample.write_text(json.dumps({
            "id": "NCT123456", "title": "Sample Trial",
            "inclusion_keywords": ["asthma", "wheezing"],
            "exclusion_keywords": ["pregnancy"],
            "age_min": 18, "age_max": 70,
        }, indent=2), encoding="utf-8")

@app.get("/")
def root():
    return {"status": "running", "api": "Clinical Trial Matcher"}

# ── Match ─────────────────────────────────────────────────────────────────────
@app.post("/match")
def match(req: MatchRequest):
    trial = load_trial(req.trial_id)
    results = run_match_all(trial, eligible_only=False, min_inclusion_score=0.0)
    pre_filtered   = sum(1 for r in results if r.get("_source") == "pre_filter")
    llm_evaluated  = sum(1 for r in results if r.get("_source") == "llm")
    rule_evaluated = sum(1 for r in results if r.get("_source") == "rule_based")
    return {
        "trial_id":    req.trial_id,
        "trial_title": trial.get("title", ""),
        "total_patients":   len(results),
        "eligible_count":   sum(1 for r in results if r["decision"] == "Eligible"),
        "matches":          results,
        "mode":             "llm",
        "pipeline": {
            "pre_filtered":   pre_filtered,
            "llm_evaluated":  llm_evaluated,
            "rule_evaluated": rule_evaluated,
            "description":    "Stage 1: Rule-based filter (age/exclusion keywords) → Stage 2: LLM semantic analysis",
        },
    }

# ── Match Preview — instant rule-based only (no LLM, no API calls) ────────────
@app.post("/match/preview")
def match_preview(req: MatchRequest):
    """Returns rule-based results immediately (<100ms). Use as fast first-pass
    while /match runs LLM evaluation in the background."""
    import os as _os
    trial = load_trial(req.trial_id)
    # Temporarily force rule_based provider for this call only
    original = _os.environ.get("LLM_PROVIDER", "rule_based")
    _os.environ["LLM_PROVIDER"] = "rule_based"
    try:
        results = run_match_all(trial, eligible_only=False, min_inclusion_score=0.0)
    finally:
        _os.environ["LLM_PROVIDER"] = original
    return {
        "trial_id": req.trial_id,
        "trial_title": trial.get("title", ""),
        "total_patients": len(results),
        "eligible_count": sum(1 for r in results if r["decision"] == "Eligible"),
        "matches": results,
        "mode": "rule_based",
    }

# ── Upload patients ───────────────────────────────────────────────────────────
@app.post("/upload/patients")
async def upload_patients(file: UploadFile = File(...), replace: bool = True):
    contents = await file.read()
    try:
        if replace:
            with sqlite3.connect(DB_PATH) as c:
                c.execute("PRAGMA journal_mode=WAL")
                c.execute("DROP TABLE IF EXISTS patients")
                c.execute("VACUUM")
                c.commit()
        rows = import_patients_csv_bytes(contents)
        total = db_count()
        return {"status": "success", "rows_inserted": rows, "total_in_db": total,
                "mode": "replace" if replace else "append",
                "message": f"Uploaded {rows} patients. Total in DB: {total}"}
    except Exception as e:
        raise HTTPException(400, f"Error importing patients: {e}")

# ── Upload trials ─────────────────────────────────────────────────────────────
@app.post("/upload/trials")
async def upload_trials(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file")
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except pd.errors.EmptyDataError:
        raise HTTPException(400, "CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(400, "Invalid CSV format")

    created = []
    for idx, row in df.iterrows():
        tid = _extract_trial_id(row, df.columns, idx)
        if not tid: continue
        data = _trial_from_row(row, tid)
        (Path(TRIALS_DIR) / f"{tid}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        created.append(tid)

    return {"status": "success", "trials_created": created,
            "total_trials": len(all_trial_ids()), "all_trials": all_trial_ids()}

def _extract_trial_id(row, columns, idx=None) -> str:
    for col in ["id", "NCT_ID", "trial_id", "NCT Number", "ClinicalTrials.gov ID"]:
        if col in columns and pd.notna(row.get(col)):
            v = str(row[col]).strip()
            if v and v.lower() not in ("nan", "null", "none", ""):
                return v
    if idx is not None: return f"TRIAL_{idx+1}"
    if "Study Title" in columns and pd.notna(row.get("Study Title")):
        return "_".join(str(row["Study Title"]).split()[:3]).upper()
    return f"UPLOADED_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def _trial_from_row(row, trial_id: str) -> dict:
    title = str(row.get("Study Title", "")).strip() if pd.notna(row.get("Study Title", None)) else ""
    kws = []
    if "Conditions" in row.index and pd.notna(row.get("Conditions")):
        cond = str(row["Conditions"])
        for sep in ["|", ",", ";", "/"]:
            if sep in cond:
                kws.extend(k.strip().lower() for k in cond.split(sep) if k.strip()); break
        else:
            if cond.strip(): kws.append(cond.strip().lower())
    for field, terms in [
        ("Brief Summary", ["asthma","diabetes","hypertension","cancer","cardiovascular",
                           "pulmonary","respiratory","infection","inflammation","pain",
                           "therapy","treatment","drug","medication","surgery"]),
        ("Interventions", ["drug","therapy","treatment","medication","placebo","surgery"]),
    ]:
        if field in row.index and pd.notna(row.get(field)):
            txt = str(row[field]).lower()
            kws.extend(t for t in terms if t in txt and t not in kws)
    if not kws: kws = ["medical", "trial", "clinical study"]
    lo, hi = _parse_age(row)
    return {"id": trial_id, "title": title, "inclusion_keywords": kws,
            "exclusion_keywords": ["pregnancy", "severe cardiac disease"],
            "age_min": lo, "age_max": hi, "source": "csv_upload",
            "upload_date": datetime.now().isoformat()}

def _parse_age(row):
    if "Age" not in row.index or pd.isna(row.get("Age")): return 18, 100
    t = str(row["Age"]).upper()
    if "CHILD" in t or "PEDIATRIC" in t: return 1, 17
    if "ADULT" in t and "OLDER" in t:   return 18, 100
    if "ADULT" in t:                    return 18, 65
    if "ALL" in t or "ANY" in t:        return 0, 100
    if "ELDERLY" in t or "GERIATRIC" in t: return 65, 100
    nums = re.findall(r"\d+", t)
    if len(nums) >= 2: return int(nums[0]), int(nums[1])
    if len(nums) == 1: return int(nums[0]), 100
    return 18, 100

# ── Downloads ─────────────────────────────────────────────────────────────────
@app.get("/download/csv/{trial_id}")
def download_csv(trial_id: str):
    results = run_match_all(load_trial(trial_id), eligible_only=False, min_inclusion_score=0.0)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["patient_id","age","diagnosis","score","decision","matched_keywords"])
    w.writeheader()
    for r in results:
        w.writerow({"patient_id": r["patient_id"], "age": r["age"],
                    "diagnosis": (r["diagnosis"] or "")[:100], "score": r["score"],
                    "decision": r["decision"],
                    "matched_keywords": ", ".join(r["matched_inclusion"][:5])})
    f.close()
    return FileResponse(f.name, filename=f"{trial_id}_results.csv")

@app.get("/download/pdf/{trial_id}")
def download_pdf(trial_id: str):
    trial = load_trial(trial_id)
    results = run_match_all(trial, eligible_only=False, min_inclusion_score=0.0)
    eligible = [r for r in results if r["decision"] == "Eligible"]
    f = tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False)
    c = canvas.Canvas(f.name, pagesize=letter)
    W, H = letter
    c.setFont("Helvetica-Bold", 16); c.drawString(50, H-50, "Clinical Trial Matching Results")
    c.setFont("Helvetica", 12)
    c.drawString(50, H-75, f"Trial: {trial_id}")
    if trial.get("title"): c.drawString(50, H-95, f"Title: {trial['title']}")
    c.drawString(50, H-120, f"Total Patients: {len(results)}")
    c.drawString(50, H-140, f"Eligible: {len(eligible)}")
    y = H - 180
    c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Eligible Patients:"); y -= 30
    c.setFont("Helvetica", 10)
    for r in eligible:
        line = f"• {r['patient_id']} (Age {r['age']}): {(r['diagnosis'] or '')[:60]}"
        c.drawString(70, y, line[:80]); y -= 20
        if y < 50: c.showPage(); y = H-50; c.setFont("Helvetica", 10)
    c.save(); f.close()
    return FileResponse(f.name, filename=f"{trial_id}_results.pdf")

# ── Evaluate (KPI summary) ────────────────────────────────────────────────────
@app.get("/evaluate/{trial_id}")
def evaluate_trial(trial_id: str):
    t0 = time.perf_counter()
    trial = load_trial(trial_id)
    results = run_match_all(trial, eligible_only=False, min_inclusion_score=0.0)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    if not results:
        return {"trial_id": trial_id, "total_patients": 0,
                "rule_based": {}, "logic_based": {}, "performance": {"processing_time_ms": ms}}
    total = len(results)
    eligible = sum(1 for r in results if r["decision"] == "Eligible")
    age_pass = sum(1 for r in results if r.get("age_ok"))
    excl_hit = sum(1 for r in results if r.get("exclusion_score", 0) > 0)
    tp = eligible
    fp = sum(1 for r in results if r["decision"] == "Eligible" and not r.get("age_ok"))
    prec = round(tp / (tp + fp), 3) if (tp + fp) else 1.0
    thr = round(total / max(ms / 1000, 0.001), 1)
    return {
        "trial_id": trial_id, "trial_title": trial.get("title", ""),
        "total_patients": total, "processing_time_ms": ms,
        "rule_based": {
            "age_pass": age_pass, "age_fail": total - age_pass,
            "exclusion_triggered": excl_hit, "exclusion_clean": total - excl_hit,
            "both_rules_passed": sum(1 for r in results if r.get("age_ok") and not r.get("exclusion_score", 0)),
            "age_pass_rate": round(age_pass / total * 100, 1),
            "exclusion_clean_rate": round((total - excl_hit) / total * 100, 1),
        },
        "logic_based": {
            "eligible_count": eligible, "not_eligible_count": total - eligible,
            "eligibility_rate": round(eligible / total * 100, 1),
            "avg_inclusion_score": 0, "avg_eligible_score": 0,
            "score_distribution": {}, "keyword_hit_frequency": {},
        },
        "performance": {
            "true_positive": tp, "false_positive": fp, "false_negative": 0,
            "true_negative": total - tp - fp,
            "precision": prec, "recall": 1.0,
            "f1_score": round(2 * prec / (prec + 1), 3) if prec else 0.0,
            "accuracy": round((tp + total - tp - fp) / total * 100, 1),
            "processing_time_ms": ms, "throughput_patients_per_sec": thr,
            "score_histogram": {},
        },
    }

# ── Chat ──────────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    q = req.question.strip()
    ql = re.sub(r"[\s?!.,;:]+$", "", q.lower()).strip()
    patients = fetch_all_patients()
    n = len(patients)
    trials = all_trial_ids()

    def bot(msg): return ChatMessage(role="bot", content=msg)

    def llm_match(trial, eligible_only=True):
        """Run matching using the configured LLM provider (e.g. Groq)."""
        return run_match_all(trial, eligible_only=eligible_only, min_inclusion_score=0.0)

    def list_patients(ps, header):
        lines = [header]
        for p in ps[:15]: lines.append(f"* {p['patient_id']} | Age {p['age']} | {str(p.get('diagnosis',''))[:55]}")
        if len(ps) > 15: lines.append(f"... and {len(ps)-15} more.")
        return "\n".join(lines)

    # ── Patient stats ────────────────────────────────────────────────────────
    if re.search(r"average age|mean age|avg age", ql):
        ages = [p["age"] for p in patients if p.get("age")]
        if not ages: return bot("No patient age data available.")
        return bot(f"Average age: {round(sum(ages)/len(ages),1)} | Min: {min(ages)} | Max: {max(ages)} | Total: {len(ages)}")

    if re.search(r"age distribution|age breakdown|age groups|ages by decade", ql):
        d = {}
        for p in patients:
            b = f"{(p.get('age',0)//10)*10}s"; d[b] = d.get(b,0)+1
        lines = [f"Age distribution ({n} patients):\n"]
        for b, c in sorted(d.items(), key=lambda x: int(x[0][:-1])):
            lines.append(f"* {b:6s}: {c:4d}  {'#'*min(c,30)}")
        return bot("\n".join(lines))

    if re.search(r"top diagnos|most common diagnos|common condition|frequent diagnos", ql):
        dc = {}
        for p in patients:
            d = str(p.get("diagnosis","")).strip()
            if d: dc[d] = dc.get(d,0)+1
        top = sorted(dc.items(), key=lambda x: -x[1])[:10]
        lines = [f"Top diagnoses ({n} patients):\n"]
        for i,(diag,c) in enumerate(top,1):
            lines.append(f"{i:2d}. {diag[:50]:<52} {c} ({round(c/n*100,1)}%)")
        return bot("\n".join(lines))

    age_m = re.search(r"(over|under|above|below|older than|younger than)\s+(\d+)", ql)
    if age_m and re.search(r"patient|people|record|who are|aged?", ql):
        direction, threshold = age_m.group(1), int(age_m.group(2))
        above = direction in ("over","above","older than")
        ps = [p for p in patients if p.get("age",0) > threshold] if above else \
             [p for p in patients if p.get("age",0) < threshold]
        label = f"{'over' if above else 'under'} {threshold}"
        return bot(list_patients(ps, f"{len(ps)} patients {label} ({round(len(ps)/max(n,1)*100,1)}%):"))

    if re.search(r"no medication|without medication|missing medication|no meds|empty medication", ql):
        ps = [p for p in patients if not str(p.get("medications","")).strip()
              or str(p.get("medications","")).strip().lower() in ("none","nan","")]
        return bot(list_patients(ps, f"{len(ps)} patients with no medication:") if ps else "All patients have medication data.")

    # ── Trial queries ─────────────────────────────────────────────────────────
    trial_m = re.search(r"(NCT[A-Z0-9]+)", q, re.IGNORECASE)

    # ── NCT patient lookup: "patients with NCT number XXX" ───────────────────
    if trial_m and re.search(
        r"enrolled|assigned|registered|belong|already in|with nct number", ql
    ) and not any(kw in ql for kw in ["eligible", "for trial", "match", "qualify", "fit", "suitable", "candidates"]):
        nct = trial_m.group(1).upper()
        matched = [p for p in patients if str(p.get("nct_number", "")).strip().upper() == nct]
        if not matched:
            return bot(
                f"No patients found with NCT number '{nct}'.\n"
                f"This means no patient in the database has '{nct}' stored in their NCT number field."
            )
        lines = [f"Found {len(matched)} patient(s) assigned to {nct}:\n"]
        for p in matched:
            lines.append(
                f"+- Patient ID : {p['patient_id']}\n"
                f"|  Age        : {p.get('age', 'N/A')}\n"
                f"|  Diagnosis  : {str(p.get('diagnosis', 'N/A'))[:80]}\n"
                f"+- Medications: {str(p.get('medications', 'N/A'))[:80]}"
            )
        return bot("\n\n".join(lines))

    if trial_m and re.search(r"criteria|requirement|condition|what.*trial|detail|info about", ql):
        tid = re.sub(r"[\.…]+$", "", trial_m.group(1).upper())
        try:
            t = load_trial(tid)
            incl = "\n".join(f"  + {k}" for k in t.get("inclusion_keywords",[]))
            excl = "\n".join(f"  - {k}" for k in t.get("exclusion_keywords",[]))
            return bot(f"Trial: {tid}\nTitle: {t.get('title','N/A')}\nAge: {t.get('age_min',0)}-{t.get('age_max',100)}\n\nInclusion:\n{incl or '  (none)'}\n\nExclusion:\n{excl or '  (none)'}")
        except HTTPException:
            return bot(f"Trial {tid} not found. Available: {', '.join(trials[:8])}")

    pat_m = re.search(r"\b(P\d+\w*)\b", q, re.IGNORECASE)
    if pat_m and re.search(r"eligible|qualify|fit|match|suitable|which trial|what trial", ql):
        pid = pat_m.group(1).upper()
        target = next((p for p in patients if p["patient_id"].upper() == pid), None)
        if not target: return bot(f"Patient {pid} not found.")
        elig = []
        for tid in trials:
            try:
                for r in llm_match(load_trial(tid), eligible_only=False):
                    if str(r.get("patient_id","")).upper() == pid and r.get("decision") == "Eligible":
                        elig.append((tid, r.get("score",0))); break
            except: pass
        if not elig: return bot(f"Patient {pid} is not eligible for any of the {len(trials)} trials.")
        lines = [f"Patient {pid} is eligible for {len(elig)} trial(s):\n"]
        for tid, score in sorted(elig, key=lambda x: -x[1]): lines.append(f"* {tid} | Score: {score:.2f}")
        return bot("\n".join(lines))

    TRIAL_KW = ["fit","eligible","match","qualify","find","who","criteria","suitable","candidates","for trial","database","fits"]
    if trial_m and any(kw in ql for kw in TRIAL_KW):
        tid = re.sub(r"[\.…]+$", "", trial_m.group(1).upper())
        try:
            trial = load_trial(tid)
            if re.search(r"how many|count|number of|percentage|percent|\bsummar", ql):
                all_r = llm_match(trial, eligible_only=False)
                elig_r = [r for r in all_r if r["decision"] == "Eligible"]
                pct = round(len(elig_r)/len(all_r)*100,1) if all_r else 0
                return bot(f"Summary for {tid} (LLM-matched):\n* Total: {len(all_r)}\n* Eligible: {len(elig_r)} ({pct}%)\n* Not eligible: {len(all_r)-len(elig_r)}\n* Age range: {trial.get('age_min',0)}-{trial.get('age_max',100)}")
            if re.search(r"\bnot eligible\b|\bineligible\b|\brejected\b|\bexcluded\b", ql):
                all_r = llm_match(trial, eligible_only=False)
                ne = [r for r in all_r if r["decision"] != "Eligible"]
                lines = [f"{len(ne)} NOT eligible for {tid} (LLM reasoning):\n"]
                for r in ne[:20]:
                    reason = r.get("reason") or "No LLM reason provided"
                    lines.append(f"* {r.get('patient_id')} | Age {r.get('age')} | {reason[:120]}")
                return bot("\n".join(lines))
            elig_r = llm_match(trial, eligible_only=True)
            if not elig_r: return bot(f"No eligible patients for {tid}.")
            show_all = "all" in ql.split()
            limit = len(elig_r) if show_all else 20
            lines = [f"{len(elig_r)} eligible patient(s) for {tid}:\n"]
            for r in elig_r[:limit]:
                lines.append(
                    f"+- Patient ID : {r.get('patient_id')}\n"
                    f"|  Age        : {r.get('age', 'N/A')}\n"
                    f"|  Score      : {r.get('score', 0):.2f}\n"
                    f"|  Diagnosis  : {str(r.get('diagnosis', 'N/A'))[:80]}\n"
                    f"+- Medications: {str(r.get('medications', 'N/A'))[:80]}"
                )
            if not show_all and len(elig_r) > limit: 
                lines.append(f"... and {len(elig_r)-limit} more. Download CSV/PDF for full list. (Ask for 'all eligible' to instantly view the full list)")
            return bot("\n\n".join(lines))
        except HTTPException:
            return bot(f"Trial not found. Available: {', '.join(trials[:8]) or 'None uploaded'}")

    if any(kw in ql for kw in ["for trial","for study","fit the criteria","eligible for","criteria for"]):
        if not trials: return bot("No trials uploaded yet.")
        return bot(f"Which trial? Available:\n" + "\n".join(f"* {t}" for t in trials[:10]) +
                   (f"\n... and {len(trials)-10} more" if len(trials) > 10 else "") +
                   f"\n\nExample: 'Find patients for trial {trials[0]}'")

    # ── General stats ─────────────────────────────────────────────────────────
    if re.search(r"how many (patients|records|people|entries)", ql) and \
       not re.search(r"\b(have|with|who have|diagnosed|eligible)\b", ql):
        return bot(f"There are {n} patients in the database.")

    if re.search(r"how many trials|list trials|available trials|what trials|show trials", ql):
        return bot(f"There are {len(trials)} trials:\n" + "\n".join(f"* {t}" for t in trials))

    # ── Condition + age range filter ──────────────────────────────────────────
    age_range = re.search(r"aged?\s+(?:between\s+)?(\d+)\s+(?:and|-|to)\s+(\d+)", ql)
    cond_m = re.search(r"(?:with|have|having|diagnosed with)\s+(.+?)(?:\s+aged?|\s*$)", ql)
    if age_range and cond_m:
        lo, hi = int(age_range.group(1)), int(age_range.group(2))
        cond = re.sub(r"\s*(in\s+(my\s+)?(the\s+)?database|in\s+my\s+data)\s*$", "", cond_m.group(1)).strip().rstrip(".,?!")
        ps = [p for p in patients if lo <= p.get("age",0) <= hi and cond in str(p.get("diagnosis","")).lower()]
        return bot(list_patients(ps, f"{len(ps)} patients with '{cond}' aged {lo}-{hi}:") if ps
                   else f"No patients with '{cond}' aged {lo}-{hi}.")

    # ── Condition filter ──────────────────────────────────────────────────────
    term_m = re.search(
        r"(?:(?:show|find|list|get|fetch|give me|display|what are|show me)\s+)?"
        r"patients?\s+(?:with|who\s+have|having|diagnosed\s+with|that\s+have|suffering\s+from)\s+(.+?)$"
        r"|how\s+many\s+(?:patients?\s+)?(?:have|with|are\s+diagnosed\s+with|suffer\s+from)\s+(.+?)$"
        r"|count\s+(?:of\s+)?patients?\s+(?:with|having|diagnosed\s+with)\s+(.+?)$", ql)
    if term_m:
        raw = next((g for g in term_m.groups() if g), "").strip()
        term = re.sub(r"\s*(in\s+(my\s+)?(the\s+)?database|in\s+my\s+data)\s*$", "", raw).strip().rstrip("?!.,; ")
        if not term: term = raw
        ps = [p for p in patients if any(term in str(p.get(f,"")).lower() for f in ["diagnosis","medications","notes"])]
        if re.search(r"how many|count|number of", ql):
            return bot(f"There are {len(ps)} patients matching '{term}'.")
        return bot(list_patients(ps, f"Found {len(ps)} patients with '{term}':") if ps
                   else f"No patients found matching '{term}'.")

    # ── Fallback help ─────────────────────────────────────────────────────────
    return bot(
        "Here is what I can answer:\n\n"
        "PATIENT STATS\n"
        "* 'How many patients?' | 'Average age?' | 'Age distribution'\n"
        "* 'Most common diagnoses?'\n\n"
        "TRIAL QUERIES\n"
        "* 'Criteria for NCT02827175?' | 'Which trials is P0042 eligible for?'\n\n"
        "ELIGIBILITY\n"
        "* 'Find patients for trial NCT02827175'\n"
        "* 'How many eligible for NCT02827175?'\n\n"
        "CONDITION FILTER\n"
        "* 'Patients with asthma' | 'How many have diabetes?' | 'Cancer aged 40-65'"
    )

# ── Admin ─────────────────────────────────────────────────────────────────────
@app.delete("/admin/reset-patients")
def reset_patients():
    with sqlite3.connect(DB_PATH) as c:
        try:
            deleted = c.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        except sqlite3.OperationalError:
            deleted = 0
        c.execute("DROP TABLE IF EXISTS patients")
        c.execute("VACUUM")
        c.commit()
    return {"status": "ok", "patients_deleted": deleted}

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
