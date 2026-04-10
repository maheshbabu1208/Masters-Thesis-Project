# Clinical Match

Clinical Match is a full‑stack demo application for matching patients to clinical trials.  
The backend is built with **FastAPI + SQLite**, and the frontend uses **React + Vite**.[web:16][web:32]

---

## Features

- Upload **patients CSV** into a SQLite database.
- Upload **trials CSV** and auto‑generate trial JSON files.
- Run **eligibility matching** for a selected trial across all patients.
- Download match results as **CSV** or **PDF**.
- Simple **chatbot** to query patient and trial counts.

---

## Tech Stack

- **Backend:** FastAPI, Uvicorn, SQLite, Pandas, Python `sqlite3`.[web:32]
- **Frontend:** React, Vite.[web:30]
- **Data storage:**
  - `data/patients.db` – patients table used for matching.
  - `data/trials/*.json` – per‑trial metadata generated from CSV.
  - `data/results` – exported CSV/PDF result files.
  - `data/uploads` – raw uploaded CSV files.

---

## Project Structure

```text
clinical_match/
├── backend/
│   ├── llm/
│   │   ├── match_logic.py      # patient–trial matching logic
│   │   └── __init__.py
│   ├── utils/
│   │   ├── pdf_export.py       # export results to PDF
│   │   ├── csv_export.py       # export results to CSV
│   │   └── __init__.py
│   └── __pycache__/
├── data/
│   ├── patients.db             # SQLite DB with patients table
│   ├── trials/                 # trial JSON files (e.g. NCT123456.json)
│   ├── results/                # generated result files
│   └── uploads/                # uploaded raw CSVs
├── ui/                         # Vite React app (src, package.json, vite.config.*)
├── main.py                     # FastAPI app entrypoint
├── config.py                   # paths: DATA_DIR, TRIALS_DIR, UPLOADS_DIR, DB_PATH
└── README.md

## backend run
cd clinical_match
python -m uvicorn main:app --reload --port 8000

## frontend run
cd clinical_match/ui
npm install
npm run dev

