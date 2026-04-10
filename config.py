
import os
from pathlib import Path

os.environ.setdefault("LLM_PROVIDER", "groq")
# GROQ_API_KEY must be set in your .env file or system environment — never hardcode secrets here
os.environ.setdefault("GROQ_MODEL",   "llama-3.1-8b-instant")  # fastest free-tier model (~5x faster than 70b)
os.environ.setdefault("OPENAI_MODEL", "gpt-4o")
os.environ.setdefault("OLLAMA_MODEL", "llama3.2:3b")
os.environ.setdefault("OLLAMA_URL",   "http://localhost:11434")
os.environ.setdefault("LLM_WORKERS",  "20")  # parallel threads for LLM matching

LLM_PROVIDER   = os.environ["LLM_PROVIDER"]
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL     = os.environ["GROQ_MODEL"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.environ["OPENAI_MODEL"]
OLLAMA_MODEL   = os.environ["OLLAMA_MODEL"]
OLLAMA_URL     = os.environ["OLLAMA_URL"]
# ──────────────────────────────────────────────────────────────────────────────

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "backend" / "data"
UPLOADS_DIR = BASE_DIR / "backend" / "uploads"
TRIALS_DIR = DATA_DIR / "trials"
RESULTS_DIR = DATA_DIR / "results"

# NEW: Directory for dynamically uploaded trials
DYNAMIC_TRIALS_DIR = DATA_DIR / "dynamic_trials"

# Database
DB_PATH = UPLOADS_DIR / "patients.db"

# Create directories if they don't exist
for directory in [DATA_DIR, UPLOADS_DIR, TRIALS_DIR, RESULTS_DIR, DYNAMIC_TRIALS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Print paths for debugging
print(f"[OK] Config loaded:")
print(f"   Base directory: {BASE_DIR}")
print(f"   Data directory: {DATA_DIR}")
print(f"   Trials directory: {TRIALS_DIR}")
print(f"   Dynamic trials directory: {DYNAMIC_TRIALS_DIR}")
print(f"   Database path: {DB_PATH}")