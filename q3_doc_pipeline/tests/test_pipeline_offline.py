"""End-to-end test of scoring / repair / cross-doc / reporting with a SIMULATED LLM.
(No API calls. The fake returns the answer key plus deliberately injected handwriting errors.)
Run:  pytest q3_doc_pipeline/tests -q -s
"""
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import cross_check  # noqa: E402
import evaluate as E  # noqa: E402
from engine import process_document  # noqa: E402
from report import write_reports  # noqa: E402
from schemas import DOC_TYPES  # noqa: E402

KEY = E.load_key(HERE / "answer_key.json")
STEM_TO_TYPE = {
    "Aadhar": "aadhaar", "ID": "pan", "ChatGPT Image May 2, 2026, 03_43_11 PM": "driving_licence",
    "ChatGPT Image May 2, 2026, 03_52_54 PM": "passport", "ECS": "nach_ecs", "Fatca": "fatca",
    "Illustration": "benefit_illustration", "Moral": "moral_hazard", "split": "multiple_policies",
    "suitability": "suitability_profiler",
}
# injected misreads on sampled (temperature>0) reads / zoom
NOISE = {
    ("ECS", "ifsc_code", 1): "SBINO227112",          # letter O for zero  -> repaired by validator, still agrees
    ("Fatca", "tin_or_pan", 0): "BPQPD305IR",        # I for 1            -> repaired by validator
    ("Illustration", "date", 1): "26/04/2025",       # one dissenting date
    ("Illustration", "date", 2): "26/04/2025",       # ...twice -> should be flagged
    ("ECS", "bank_account_number", 2): "31004258972",
}


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def json(self, contents, *, tag=None, **kw):
        self.calls += 1
        kind, stem = tag[0], tag[1]
        file = next(f for f in KEY if Path(f).stem == stem)
        dt = STEM_TO_TYPE[stem]
        prompt = contents[-1] if isinstance(contents[-1], str) else ""
        if kind == "classify":
            return {"doc_type": dt, "confidence": 0.97, "reason": "fake"}
        if kind == "locate":
            names = re.findall(r'- "(\w+)":', prompt)
            return {"boxes": {n: [300, 100, 340, 500] for n in names}}
        if kind == "extract":
            i = tag[2]
            out = {}
            for f in DOC_TYPES[dt]["fields"]:
                v = NOISE.get((stem, f, i), KEY[file].get(f))
                out[f] = {"value": v, "confidence": 0.9 if (stem, f, i) not in NOISE else 0.6}
            return {"fields": out}
        if kind == "zoom":
            names = re.findall(r'- "(\w+)":', prompt)
            return {"fields": {n: {"value": KEY[file].get(n), "confidence": 0.95} for n in names}}
        raise AssertionError(kind)


def run_all(tmp):
    llm = FakeLLM()
    results = []
    for p in sorted((HERE / "data").iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            results.append(process_document(llm, p, p.stem, n_samples=3, use_zoom=True))
    by = {r["document_type"]["value"]: r for r in results}
    findings = cross_check.run(by)
    write_reports(results, findings, Path(tmp), 0.80)
    return results, findings, llm


def test_pipeline_logic():
    with tempfile.TemporaryDirectory() as tmp:
        results, findings, llm = run_all(tmp)
        by = {r["file"]: r for r in results}
        ecs, fatca, ill = by["ECS.jpeg"]["fields"], by["Fatca.jpeg"]["fields"], by["Illustration.jpeg"]["fields"]

        # validator repair makes O/0 and I/1 misreads agree with the true value
        assert ecs["ifsc_code"]["value"] == "SBIN0227112"
        assert fatca["tin_or_pan"]["value"] == "BPQPD3051R"
        # a field with a dissenting minority is still resolved correctly by the zoom vote ...
        assert ill["date"]["value"] == "26/04/2026"
        # ... and the disagreement lowers confidence below that of a unanimous field
        assert ill["date"]["confidence"] < ill["application_number"]["confidence"]
        assert ill["date"]["needs_review"], "2 of 4 reads dissent: must be flagged even though 3 other forms agree"
        assert "corroborated" not in " ".join(ill["date"].get("notes", []))
        assert ecs["bank_account_number"]["confidence"] < ecs["bank_name"]["confidence"]

        # cross-document rules
        groups = {f["group"] for f in findings}
        assert "pan_vs_tin" in groups                      # ABCDE1234F vs BPQPD3051R
        assert "fathers_name" in groups                    # Kumar vs Arjun Das Kumar (partial)
        assert fatca["tin_or_pan"]["needs_review"] and by["ID.png"]["fields"]["pan_number"]["needs_review"]
        # passport MRZ check digits fail on the sample -> flagged
        mrz = by["ChatGPT Image May 2, 2026, 03_52_54 PM.png"]["fields"]["mrz_line_2"]
        assert mrz["validator"]["status"] == "warn" and "dob" in " ".join(mrz["validator"]["notes"])

        rows = E.compare(results, KEY)
        assert all(r["correct"] for r in rows), [r for r in rows if not r["correct"]]
        print("\nLLM calls simulated:", llm.calls)
        print(Path(tmp, "flag_report.md").read_text(encoding="utf-8")[:3500])


if __name__ == "__main__":
    test_pipeline_logic()
