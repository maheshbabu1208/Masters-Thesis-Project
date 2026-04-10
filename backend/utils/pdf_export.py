from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union
import os

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

PathLike = Union[str, os.PathLike]


def generate_pdf(trial: Dict[str, Any], results: List[Dict[str, Any]], out_path: PathLike) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"Trial: {trial.get('id', '')} - {trial.get('title', '')}")
    y -= 22

    c.setFont("Helvetica", 10)
    c.drawString(
        margin,
        y,
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
    )
    y -= 18

    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Matches:")
    y -= 14
    c.setFont("Helvetica", 10)

    for r in results:
        if y < margin + 80:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)

        patient_id = r.get("patient_id", "")
        age = r.get("age", "")
        score = r.get("score", "")
        decision = r.get("decision", "")

        c.drawString(
            margin,
            y,
            f"Patient {patient_id} | Age: {age} | Score: {score} | {decision}",
        )
        y -= 14

        incl = ", ".join(r.get("matched_inclusion", []) or []) or "None"
        excl = ", ".join(r.get("matched_exclusion", []) or []) or "None"

        c.drawString(margin + 12, y, f"Incl: {incl}")
        y -= 12
        c.drawString(margin + 12, y, f"Excl: {excl}")
        y -= 16

    c.save()