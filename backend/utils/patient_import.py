import csv
import io
import sqlite3
import uuid
from typing import Dict, Set, Tuple

from config import DB_PATH


def ensure_patients_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            nct_number TEXT,
            age INTEGER,
            diagnosis TEXT,
            medications TEXT,
            notes TEXT
        )
        """
    )
    conn.commit()


def normalize_patient_row(
    row: Dict[str, str], seen_ids: Set[str], row_index: int
) -> Tuple[str, str, int, str, str, str]:
    """
    Map CSV headers to DB columns.

    CSV headers (your file):
      patient_id, patient_name, age, nct_number, diagnosis, Disease, medication
    DB columns:
      patient_id, nct_number, age, diagnosis, medications, notes

    If patient_id is blank or already seen in this upload, a unique ID is
    generated so that every CSV row becomes its own DB record.
    """
    patient_id = (row.get("patient_id") or "").strip()

    # Generate a unique ID when blank or a non-value sentinel
    if not patient_id or patient_id.lower() in ("nan", "null", "none"):
        patient_id = f"AUTO_{row_index:06d}_{uuid.uuid4().hex[:6]}"
    elif patient_id in seen_ids:
        # Same ID appears more than once in this CSV → keep rows separate
        patient_id = f"{patient_id}_DUP_{row_index:04d}"

    seen_ids.add(patient_id)

    nct_number = (row.get("nct_number") or "").strip()

    try:
        age = int((row.get("age") or "0").strip())
    except Exception:
        age = 0

    diagnosis   = (row.get("diagnosis") or "").strip()
    medications = (row.get("medication") or "").strip()

    patient_name = (row.get("patient_name") or "").strip()
    disease      = (row.get("Disease") or "").strip()
    notes        = "; ".join([x for x in [patient_name, disease] if x])

    return (patient_id, nct_number, age, diagnosis, medications, notes)


def import_patients_csv_bytes(contents: bytes) -> int:
    text   = contents.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    seen_ids: Set[str] = set()
    rows = []
    for i, row in enumerate(reader):
        rows.append(normalize_patient_row(row, seen_ids, i))

    print(f"[IMPORT] CSV rows read: {len(rows)} | unique IDs in file: {len(seen_ids)}")

    with sqlite3.connect(DB_PATH) as conn:
        ensure_patients_table(conn)
        # WAL mode + relaxed sync = much faster bulk writes
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO patients
                (patient_id, nct_number, age, diagnosis, medications, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        total_in_db = cur.execute("SELECT COUNT(*) FROM patients").fetchone()[0]

    print(f"[IMPORT] Done. Rows in file: {len(rows)} | Total patients in DB now: {total_in_db}")
    return len(rows)
