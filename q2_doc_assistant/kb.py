"""Knowledge base: scanned PDFs -> sections (Gemini vision, or Tesseract fallback) -> chunks -> TF-IDF index.

The transcribed sections are cached in kb/sections.json so you can open that file, fix any OCR slip, and
re-run without spending API quota. Every chunk keeps (document, page, section heading) for citations.
"""
import io
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SECTIONS_JSON = HERE / "kb" / "sections.json"
DOCS_DIR = HERE / "docs"
# optional extra material: the printed text of the HDFC forms supplied for Q3
Q3_FORMS = ["ECS.jpeg", "Fatca.jpeg", "Illustration.jpeg", "Moral.jpeg", "split.jpeg", "suitability.jpeg"]
Q3_DATA = HERE.parent / "q3_doc_pipeline" / "data"

TRANSCRIBE_PROMPT = """This is one page of an HDFC Life Insurance form (scanned). Transcribe it into logical sections.
Return JSON: {"sections": [{"heading": "<short heading as printed, or a short descriptive one such as 'Details of the Assignee'>",
"text": "<full text of that section>"}]}
Rules: keep numbered clauses/notes numbered exactly as printed; include printed field labels; write handwritten entries as
[handwritten: ...] and unreadable bits as [illegible]; skip pure decoration; do not summarise, interpret or add commentary;
keep the original wording."""


@dataclass
class Chunk:
    id: int
    doc: str
    page: int
    section: str
    text: str

    @property
    def label(self) -> str:
        return f"{self.doc} · p.{self.page} · {self.section}"


# ------------------------------------------------------------------ page rendering
def render_pages(path: Path, dpi=200):
    from PIL import Image

    if path.suffix.lower() == ".pdf":
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        doc = fitz.open(path)
        for i, page in enumerate(doc, 1):
            pix = page.get_pixmap(dpi=dpi)
            yield i, Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    else:
        yield 1, Image.open(path).convert("RGB")


def _jpeg(im, max_side=2200):
    w, h = im.size
    s = max_side / max(w, h)
    if s < 1:
        im = im.resize((int(w * s), int(h * s)))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=92)
    return b.getvalue()


# ------------------------------------------------------------------ transcription
def transcribe_vision(llm, im, doc, page):
    from common.llm import image_part

    out = llm.json([image_part(_jpeg(im)), TRANSCRIBE_PROMPT], temperature=0.0)
    secs = out.get("sections", []) if isinstance(out, dict) else out
    return [{"heading": (s.get("heading") or f"Page {page}").strip(), "text": (s.get("text") or "").strip()}
            for s in secs if (s.get("text") or "").strip()]


def transcribe_tesseract(im, doc, page):
    """EMERGENCY fallback only. Tesseract cannot read these scans reliably, so headings and text come out garbled.
    Use --mode vision for anything you intend to cite. Heuristic sectioning: short ALL-CAPS / colon-ended line = heading."""
    import pytesseract

    print("  [warn] Tesseract mode gives poor transcription on these scans; prefer --mode vision")

    text = pytesseract.image_to_string(im)
    secs, cur_h, cur = [], f"Page {page}", []
    for line in [l.strip() for l in text.splitlines()]:
        if not line:
            continue
        is_head = len(line) < 60 and (line.isupper() or line.endswith(":")) and len(line) > 4
        if is_head and cur:
            secs.append({"heading": cur_h, "text": " ".join(cur)})
            cur_h, cur = line.rstrip(":").title(), []
        elif is_head:
            cur_h = line.rstrip(":").title()
        else:
            cur.append(line)
    if cur:
        secs.append({"heading": cur_h, "text": " ".join(cur)})
    return secs


def ingest(llm=None, mode="vision", extra_forms=False, fresh=False, out=SECTIONS_JSON, docs_dir=DOCS_DIR):
    """Transcribe all docs -> list of {doc, page, heading, text}; cached."""
    out = Path(out)
    if out.exists() and not fresh:
        return json.loads(out.read_text(encoding="utf-8"))
    paths = sorted(p for p in Path(docs_dir).iterdir() if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"})
    if extra_forms:
        paths += [Q3_DATA / n for n in Q3_FORMS if (Q3_DATA / n).exists()]
    rows = []
    for p in paths:
        for page, im in render_pages(p):
            print(f"  transcribing {p.name} p.{page} ({mode})")
            secs = transcribe_vision(llm, im, p.name, page) if mode == "vision" else transcribe_tesseract(im, p.name, page)
            rows += [dict(doc=p.name, page=page, heading=s["heading"], text=s["text"]) for s in secs]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return rows


# ------------------------------------------------------------------ chunking + retrieval
def make_chunks(rows, max_chars=900):
    chunks = []
    for r in rows:
        text = re.sub(r"[ \t]+", " ", r["text"]).strip()
        parts = [text] if len(text) <= max_chars else _split(text, max_chars)
        for k, part in enumerate(parts):
            head = r["heading"]
            nums = re.findall(r"(?:^|\s)(\d{1,2})[.)]\s", part)
            if len(parts) > 1:
                head += f" (items {nums[0]}-{nums[-1]})" if len(nums) > 1 else (f" (item {nums[0]})" if nums else f" (part {k + 1})")
            chunks.append(Chunk(len(chunks), r["doc"], r["page"], head, part))
    return chunks


def _split(text, max_chars):
    pieces = re.split(r"(?<=[.;])\s+(?=(?:\d{1,2}[.)]|[A-Z]))", text)
    out, cur = [], ""
    for p in pieces:
        if len(cur) + len(p) > max_chars and cur:
            out.append(cur.strip())
            cur = ""
        cur += p + " "
    if cur.strip():
        out.append(cur.strip())
    return out


class KnowledgeBase:
    """Hybrid lexical retrieval: word 1-2grams + character 3-5grams (handles 'service'/'servicing', OCR slips).
    Scores are only used for ranking. They cannot separate answerable from unanswerable questions (a question that merely
    shares words still scores), so relevance is judged by the LLM in the answer step."""

    def __init__(self, rows):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.rows = rows
        self.chunks = make_chunks(rows)
        # headings are repeated so a query matching a section title ranks that section higher
        docs = [f"{c.section}. {c.section}. {c.text}" for c in self.chunks]
        self.wvec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.WX, self.CX = self.wvec.fit_transform(docs), self.cvec.fit_transform(docs)

    def search(self, query, k=6, min_score=0.03):
        from sklearn.metrics.pairwise import linear_kernel

        sims = 0.5 * linear_kernel(self.wvec.transform([query]), self.WX).ravel() \
            + 0.5 * linear_kernel(self.cvec.transform([query]), self.CX).ravel()
        order = sims.argsort()[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in order if sims[i] >= min_score]

    def toc(self):
        seen, lines = {}, []
        for c in self.chunks:
            seen.setdefault(c.doc, [])
            h = re.sub(r"\s*\((?:items?|part)[^)]*\)", "", c.section)
            if h not in seen[c.doc]:
                seen[c.doc].append(h)
        for d, hs in seen.items():
            lines.append(f"{d}: " + "; ".join(hs))
        return "\n".join(lines)


def load_kb(llm=None, mode="vision", extra_forms=False, fresh=False):
    return KnowledgeBase(ingest(llm, mode, extra_forms, fresh))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the knowledge base (cached in kb/sections.json)")
    ap.add_argument("--mode", choices=["vision", "tesseract"], default="vision", help="tesseract = emergency fallback, poor quality")
    ap.add_argument("--extra-forms", action="store_true", help="also ingest the six HDFC forms from the Q3 sample files")
    ap.add_argument("--fresh", action="store_true", help="re-transcribe even if cached")
    a = ap.parse_args()
    llm = None
    if a.mode == "vision":
        from common.llm import LLM

        llm = LLM()
    kb = load_kb(llm, a.mode, a.extra_forms, a.fresh)
    print(f"{len(kb.chunks)} chunks\n{kb.toc()}")
