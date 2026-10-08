"""Score extractions against an answer key and sweep the flagging threshold.

    python evaluate.py                      # uses out/all_extractions.json + answer_key.json
    python evaluate.py --key my_key.json
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import validators as V  # noqa: E402
from schemas import DOC_TYPES  # noqa: E402

KEY_FIELDS = {"ifsc_code", "tin_or_pan", "bank_account_number", "date", "place", "place_of_birth"}


def load_key(path):
    k = json.loads(Path(path).read_text(encoding="utf-8"))
    return {f: v for f, v in k.items() if not f.startswith("_")}


def compare(results, key):
    rows = []
    for r in results:
        gt = key.get(r["file"])
        if not gt:
            continue
        dt = r["document_type"]["value"]
        for field, truth in gt.items():
            spec = DOC_TYPES.get(dt, {}).get("fields", {}).get(field)
            got = r.get("fields", {}).get(field)
            kind = spec["kind"] if spec else "text"
            ok = bool(got and got.get("value") and V.normalize(kind, got["value"]) == V.normalize(kind, truth))
            rows.append(dict(file=r["file"], doc=dt, field=field, correct=ok, conf=(got or {}).get("confidence", 0.0),
                             handwritten=bool(spec and DOC_TYPES[dt]["handwritten"]), truth=truth,
                             got=(got or {}).get("value"), flagged=(got or {}).get("needs_review", True)))
    return rows


def sweep(rows, thresholds=(0.6, 0.7, 0.8, 0.85, 0.9)):
    out = []
    wrong = [r for r in rows if not r["correct"]]
    right = [r for r in rows if r["correct"]]
    for t in thresholds:
        caught = sum(1 for r in wrong if r["conf"] < t)
        burden = sum(1 for r in right if r["conf"] < t)
        out.append(dict(threshold=t, errors_caught=caught, errors_total=len(wrong),
                        correct_flagged=burden, correct_total=len(right),
                        review_load=sum(1 for r in rows if r["conf"] < t)))
    return out


def summary_text(rows):
    def acc(rs):
        return f"{sum(r['correct'] for r in rs)}/{len(rs)} ({(sum(r['correct'] for r in rs) / len(rs) * 100 if rs else 0):.0f}%)"

    lines = ["Field-level accuracy vs answer key", "-" * 40,
             f"overall            {acc(rows)}",
             f"printed fields     {acc([r for r in rows if not r['handwritten']])}",
             f"handwritten fields {acc([r for r in rows if r['handwritten']])}",
             f"high-value handwritten (IFSC/TIN/account/date/place) {acc([r for r in rows if r['handwritten'] and r['field'] in KEY_FIELDS])}",
             "", "Per document:"]
    for f in sorted({r['file'] for r in rows}):
        rs = [r for r in rows if r['file'] == f]
        lines.append(f"  {f[:44]:44s} {acc(rs)}")
    wrong = [r for r in rows if not r["correct"]]
    if wrong:
        lines += ["", "Incorrect / missing fields:"]
        lines += [f"  {r['file'][:30]:30s} {r['field']:24s} got={r['got']!r} truth={r['truth']!r} conf={r['conf']}" for r in wrong]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "out" / "all_extractions.json"))
    ap.add_argument("--key", default=str(HERE / "answer_key.json"))
    a = ap.parse_args()
    res = json.loads(Path(a.results).read_text(encoding="utf-8"))
    rows = compare(res, load_key(a.key))
    print(summary_text(rows))
    print("\nThreshold sweep (errors_caught = wrong fields that would be sent to a human):")
    for s in sweep(rows):
        print(f"  t={s['threshold']:.2f}  errors caught {s['errors_caught']}/{s['errors_total']}  "
              f"correct fields needlessly flagged {s['correct_flagged']}/{s['correct_total']}")
