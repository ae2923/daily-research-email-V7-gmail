"""
run_briefing.py — Runs gather → analyze → email
=================================================
Point your GitHub Actions workflow at this file.
"""

from pathlib import Path
from gather_news import run as gather
from analyze_briefing import run as analyze

if __name__ == "__main__":
    print("=" * 50, flush=True)
    print("STEP 1: Gathering news", flush=True)
    print("=" * 50, flush=True)
    facts_path = gather()

    print("\n" + "=" * 50, flush=True)
    print("STEP 2: Analyzing + sending email", flush=True)
    print("=" * 50, flush=True)
    analyze(Path(facts_path))
