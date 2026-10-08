"""Run everything needed for the submission, one step after another, and keep going if a step fails.

    python run_all.py            # doctor -> Q3 pipeline -> Q3 evaluate -> Q1 demo -> Q2 KB + demo
    python run_all.py --skip q1  # skip a part (q3 / q1 / q2 / doctor)

Every LLM call is cached for Q3, so re-running after a failure only repeats what is missing.
Outputs: q3_doc_pipeline/out/, q1_inventory_agent/demo_transcript.md, q2_doc_assistant/demo_transcript.md
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
STEPS = [
    ("doctor", "provider health check", [PY, "common/doctor.py"]),
    ("q3", "Q3 pipeline (10 documents)", [PY, "q3_doc_pipeline/run_pipeline.py"]),
    ("q3", "Q3 evaluate vs answer key", [PY, "q3_doc_pipeline/evaluate.py"]),
    ("q1", "Q1 scripted demo", [PY, "q1_inventory_agent/run_demo.py"]),
    ("q2", "Q2 build knowledge base (vision OCR)", [PY, "q2_doc_assistant/kb.py"]),
    ("q2", "Q2 scripted 13-turn demo", [PY, "q2_doc_assistant/run_demo.py"]),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], choices=["doctor", "q1", "q2", "q3"])
    a = ap.parse_args()
    summary = []
    for group, name, cmd in STEPS:
        if group in a.skip:
            continue
        print(f"\n{'=' * 70}\n== {name}\n{'=' * 70}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=ROOT)
        summary.append((name, rc == 0, time.time() - t0))
    print(f"\n{'=' * 70}\nSUMMARY")
    for name, ok, dt in summary:
        print(f"  {'OK  ' if ok else 'FAIL'} {name} ({dt:.0f}s)")
    print("\nIf something failed, read the message above it, fix the cause (usually a key or a busy provider) and run again:\n"
          "finished Q3 calls are cached, so a rerun is quick.")
    sys.exit(0 if all(ok for _, ok, _ in summary) else 1)
