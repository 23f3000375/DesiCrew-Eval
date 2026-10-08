"""Run the document pipeline.

    python run_pipeline.py                       # all images in ./data
    python run_pipeline.py --only ECS            # only files whose name contains 'ECS'
    python run_pipeline.py --threshold 0.85
    python run_pipeline.py --fresh               # ignore cached LLM responses

LLM responses are cached under out/raw_cache so re-running (e.g. to tweak scoring or the
report) costs no API quota.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import cross_check  # noqa: E402
from engine import process_document  # noqa: E402
from report import write_reports  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg"}


class CachedLLM:
    """Wraps LLM; caches parsed JSON by (tag, temperature) and remembers which model produced each answer."""

    def __init__(self, llm, cache_dir: Path, fresh=False):
        self.llm, self.dir, self.fresh = llm, cache_dir, fresh
        self.dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.used = set()

    def json(self, contents, *, tag=None, **kw):
        key = hashlib.md5(json.dumps([tag, kw.get("temperature")], default=str).encode()).hexdigest()
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(map(str, tag or ["x"])))
        f = self.dir / f"{name}_{key[:6]}.json"
        if f.exists() and not self.fresh:
            try:
                blob = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(blob, dict) and "_model" in blob and "out" in blob:
                    self.used.add(blob["_model"])
                    return blob["out"]
            except (json.JSONDecodeError, OSError):
                pass  # corrupt cache entry: just call again
        self.calls += 1
        out = self.llm.json(contents, tag=tag, **kw)
        model = self.llm.model
        self.used.add(model)
        f.write_text(json.dumps({"_model": model, "out": out}, indent=1), encoding="utf-8")
        return out

    @property
    def model(self):
        return self.llm.model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--samples", type=int, default=3, help="full-page reads per handwritten document")
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--model", default=None, help="put this model first, e.g. groq:llama-3.3-70b-versatile")
    args = ap.parse_args()

    from common.llm import LLM

    out = Path(args.out)
    (out / "extractions").mkdir(parents=True, exist_ok=True)
    llm = CachedLLM(LLM(model=args.model), out / "raw_cache", fresh=args.fresh)

    files = sorted(p for p in Path(args.data).iterdir() if p.suffix.lower() in IMG_EXT)
    if args.only:
        files = [p for p in files if args.only.lower() in p.name.lower()]
    print(f"Processing {len(files)} documents. LLM chain: {' > '.join(llm.llm.order)} (models discovered on first call)")

    results, failed = [], []
    for p in files:
        print(f"-> {p.name}")
        llm.used = set()
        try:
            r = process_document(llm, p, p.stem, n_samples=args.samples, use_zoom=not args.no_zoom)
        except Exception as e:  # noqa: BLE001 - one bad document must not lose the whole run; cached calls make a rerun cheap
            print(f"   !! failed: {str(e)[:300]}\n   (progress is cached; re-run the same command to retry just what is missing)")
            failed.append(p.name)
            results.append(dict(file=p.name, document_type=dict(value="unknown", confidence=0.0, reason="processing failed"),
                                handwritten=None, fields={}, flags=[f"processing failed: {str(e)[:200]}"]))
            continue
        r["models_used"] = sorted(llm.used)
        print(f"   classified as {r['document_type']['value']} ({r['document_type']['confidence']:.2f}); models: {', '.join(r['models_used'])}")
        results.append(r)
    by_type = {r["document_type"]["value"]: r for r in results if r["document_type"]["value"] != "unknown"}
    findings = cross_check.run(by_type)
    write_reports(results, findings, out, args.threshold)
    print(f"Done. {llm.calls} LLM calls made this run. Calls per model: {dict(llm.llm.stats)}")
    if failed:
        print(f"WARNING: {len(failed)} document(s) failed: {failed}. Re-run to retry them (everything else is cached).")
    print(f"See {out}/flag_report.md")


if __name__ == "__main__":
    main()
