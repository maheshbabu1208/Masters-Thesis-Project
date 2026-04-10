from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Union
import os

PathLike = Union[str, os.PathLike]


def generate_csv(
    trial: Dict[str, Any],
    results: List[Dict[str, Any]],
    out_path: PathLike,
) -> None:
    """
    Write matching results to a CSV file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        # Nothing to write, but still create an empty file with header
        fieldnames = ["patient_id", "nct_number", "age", "diagnosis",
                      "medications", "notes", "score", "decision",
                      "matched_inclusion", "matched_exclusion", "age_ok"]
    else:
        # Use keys from the first result as header
        fieldnames = list(results[0].keys())

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Convert list fields to comma-separated strings
            row = dict(row)
            if isinstance(row.get("matched_inclusion"), list):
                row["matched_inclusion"] = ", ".join(row["matched_inclusion"])
            if isinstance(row.get("matched_exclusion"), list):
                row["matched_exclusion"] = ", ".join(row["matched_exclusion"])
            writer.writerow(row)