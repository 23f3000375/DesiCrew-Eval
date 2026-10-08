"""Write per-document JSON, the combined JSON, and the flagging report."""
import json
from pathlib import Path

import evaluate as E


def finalize_flags(results, threshold):
    for r in results:
        r["needs_human_review"] = False
        doc_flags = list(r.get("flags", []))
        if r["document_type"]["confidence"] < threshold:
            doc_flags.append(f"classification confidence {r['document_type']['confidence']:.2f} below threshold")
        nflag = 0
        for f, x in r.get("fields", {}).items():
            reasons = []
            if x["value"] is None:
                reasons.append("missing")
            if x["confidence"] < threshold:
                reasons.append("low_confidence")
            if x["validator"]["status"] == "invalid":
                reasons.append("validator_failed")
            if x.get("cross_doc_conflict"):
                reasons.append("cross_document_conflict")
            x["needs_review"] = bool(reasons)
            x["review_reasons"] = reasons
            nflag += bool(reasons)
        r["flags"] = doc_flags
        r["needs_human_review"] = bool(doc_flags) or nflag > 0
        r["fields_flagged"] = nflag


RATIONALE = """\
**Threshold: {t:.2f}** (per-field confidence; configurable with `--threshold`).

Why 0.80:
- A field scores `0.5*model_confidence + 0.5*agreement`, multiplied by a validator factor (valid 1.0, warning 0.93, failed 0.6).
  A value read identically in every sample, with model confidence >= 0.9 and a passing validator, lands at >= 0.95.
- One dissenting read out of four (3 full-page + 1 zoomed crop weighted x2) still scores about 0.85: the zoom read and two full
  reads agree, and the dissent is recorded in the field's details. Two dissenting full-page reads, or the zoom read
  disagreeing with the full-page reads, drop agreement to 0.6 or below and the field falls under 0.80.
  A failed validator (e.g. an IFSC that does not match `AAAA0xxxxxx`) multiplies the score by 0.6 and always flags.
- Cross-document corroboration (+0.03 per agreeing document, max +0.09) is applied only to fields whose own reads were
  unanimous, so it can never hide within-document disagreement.
- Cost asymmetry: a wrong IFSC / account number / TIN breaks NACH debits and onboarding, while a human check takes seconds,
  so the threshold errs on the side of flagging. Printed IDs corroborated by Tesseract score >= 0.9 and normally pass.
- The sweep below (computed against the answer key) shows the trade-off between errors caught and review load.
"""


def write_reports(results, findings, out: Path, threshold: float):
    finalize_flags(results, threshold)
    (out / "extractions").mkdir(parents=True, exist_ok=True)
    for r in results:
        (out / "extractions" / (Path(r["file"]).stem.replace(" ", "_") + ".json")).write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "all_extractions.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    flagged = []
    for r in results:
        for f, x in r.get("fields", {}).items():
            if x["needs_review"]:
                flagged.append(dict(file=r["file"], document_type=r["document_type"]["value"], field=f, value=x["value"],
                                    confidence=x["confidence"], reasons=x["review_reasons"], details=x["flags"]))
    (out / "flag_report.json").write_text(json.dumps(dict(threshold=threshold, flagged_fields=flagged,
                                                          cross_document_findings=findings), indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Human-review flag report", "", RATIONALE.format(t=threshold), ""]
    key_path = Path(__file__).with_name("answer_key.json")
    if key_path.exists():
        rows = E.compare(results, E.load_key(key_path))
        md += ["## Threshold sweep vs draft answer key", "", "| threshold | errors caught | correct fields flagged |", "|---|---|---|"]
        for s in E.sweep(rows):
            md.append(f"| {s['threshold']:.2f} | {s['errors_caught']}/{s['errors_total']} | {s['correct_flagged']}/{s['correct_total']} |")
        md += ["", "```", E.summary_text(rows), "```", ""]
    md += ["## Documents", "", "| file | type | class. conf | fields flagged | review? |", "|---|---|---|---|---|"]
    for r in results:
        n = len(r.get("fields", {}))
        md.append(f"| {r['file']} | {r['document_type']['value']} | {r['document_type']['confidence']:.2f} | {r['fields_flagged']}/{n} | {'YES' if r['needs_human_review'] else 'no'} |")
    md += ["", f"## Flagged fields ({len(flagged)})", "", "| document | field | value | conf | reasons |", "|---|---|---|---|---|"]
    for x in flagged:
        why = "; ".join(x["details"]) or ", ".join(x["reasons"])
        md.append(f"| {x['document_type']} | {x['field']} | `{x['value']}` | {x['confidence']:.2f} | {x['reasons'][0] if len(x['reasons']) == 1 else ', '.join(x['reasons'])}: {why} |")
    if findings:
        md += ["", "## Cross-document findings", ""]
        md += [f"- **{f['severity']}** ({f['group']}): {f.get('detail', '')} {f.get('values', '')}" for f in findings]
    doc_flags = [(r["file"], f) for r in results for f in r.get("flags", [])]
    if doc_flags:
        md += ["", "## Document-level flags", ""] + [f"- {a}: {b}" for a, b in doc_flags]
    (out / "flag_report.md").write_text("\n".join(md), encoding="utf-8")
