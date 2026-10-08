"""Classification + extraction + per-field confidence scoring.

Confidence for a field is built from independent signals instead of trusting the model's
self-reported number (LLMs are over-confident on handwriting):

  model_conf   weighted mean of the confidences the model reported for the winning value
  agreement    share of "votes" that agree on the winning value
                 handwritten: 3 full-page reads (1 greedy + 2 sampled) + 1 zoomed-crop read (weight 2)
                 printed:     1 LLM read cross-checked against Tesseract OCR text
  validator    format/checksum check (PAN structure, IFSC pattern, MRZ check digits, ...)

    base  = 0.5 * model_conf + 0.5 * agreement
    final = base * validator_factor            (valid/na 1.0, warn 0.93, invalid 0.6)

Cross-document adjustments are applied afterwards in cross_check.py.
"""
import io
import re
from collections import defaultdict

from PIL import Image, ImageOps

import validators as V
from schemas import DOC_TYPES

MAX_SIDE = 2000
VALIDATOR_FACTOR = {"valid": 1.0, "na": 1.0, "warn": 0.93, "invalid": 0.6}
ZOOM_WEIGHT = 2.0

SYSTEM = (
    "You are a meticulous document-digitisation engine for insurance onboarding. "
    "You transcribe exactly what is written on the page and never guess, autocorrect or invent values. "
    "Reply with JSON only."
)


# ------------------------------------------------------------------ images
def load_image(path, enhance=False):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = im.size
    scale = MAX_SIDE / max(w, h)
    if scale < 1:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    if enhance:  # mild contrast stretch helps faint pen strokes on scans/photos
        im = ImageOps.autocontrast(im, cutoff=1)
    return im


def to_jpeg(im, quality=92):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def crop_box(im, box, pad_y=0.35, pad_x=0.06, min_width=1200):
    """box = [ymin, xmin, ymax, xmax] on a 0-1000 scale (Gemini convention)."""
    ymin, xmin, ymax, xmax = [max(0, min(1000, float(v))) for v in box]
    w, h = im.size
    bh, bw = (ymax - ymin) / 1000 * h, (xmax - xmin) / 1000 * w
    y0 = max(0, ymin / 1000 * h - pad_y * bh)
    y1 = min(h, ymax / 1000 * h + pad_y * bh)
    x0 = max(0, xmin / 1000 * w - pad_x * w * 0.3 - 0.15 * bw)
    x1 = min(w, xmax / 1000 * w + pad_x * w * 0.3 + 0.15 * bw)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    c = im.crop((int(x0), int(y0), int(x1), int(y1)))
    if c.width < min_width:
        f = min_width / c.width
        c = c.resize((int(c.width * f), int(c.height * f)), Image.LANCZOS)
    return ImageOps.autocontrast(c, cutoff=1)


# ------------------------------------------------------------------ prompts
def classify_prompt():
    lines = [f'- "{k}": {v["label"]} ({v["cues"]})' for k, v in DOC_TYPES.items()]
    return (
        "Classify this document image into exactly one of these types:\n" + "\n".join(lines) +
        '\n- "unknown": anything else\n\n'
        'Return JSON: {"doc_type": "<key>", "confidence": <0-1>, "reason": "<one short sentence>"}'
    )


def extract_prompt(doc_type):
    spec = DOC_TYPES[doc_type]
    fields = "\n".join(f'- "{k}": {f["desc"]}' for k, f in spec["fields"].items())
    hw = ""
    if spec["handwritten"]:
        hw = (
            "\nThis is a HANDWRITTEN form. Rules:\n"
            "* Read ONLY the filled-in handwritten values, never the printed labels or placeholders such as DD/MM/YYYY.\n"
            "* Transcribe character by character. Do not normalise, expand or correct. Keep dates as written.\n"
            "* Digits vs letters: account / policy / application numbers are digits only; an IFSC is 4 letters, "
            "then the digit 0, then 6 letters/digits; a PAN/TIN is 5 letters, 4 digits, 1 letter.\n"
            "* If a character is ambiguous (1/7, 0/6, 2/Z, O/0, I/1), choose the most likely and list the alternative "
            "full values under \"alternatives\", and lower the confidence.\n"
            "* Checkbox fields: return the label of the TICKED box only. Struck-out options are NOT selected.\n"
        )
    return (
        f"Document type: {spec['label']}.\n"
        f"Extract these fields:\n{fields}\n{hw}\n"
        "If a field is blank, missing or illegible return value null with confidence 0. "
        "confidence = your honest probability (0-1) that the value is correct character-for-character.\n"
        'Return JSON: {"fields": {"<field>": {"value": <string|null>, "confidence": <0-1>, '
        '"alternatives": [<strings>], "note": "<optional>"}}}'
    )


def locate_prompt(doc_type, fields):
    desc = "\n".join(f'- "{k}": {DOC_TYPES[doc_type]["fields"][k]["desc"]}' for k in fields)
    return (
        f"Document type: {DOC_TYPES[doc_type]['label']}.\n"
        "For each field below, return a tight bounding box around the filled-in VALUE only (not the printed label). "
        "Use the format [ymin, xmin, ymax, xmax] on a 0-1000 scale relative to the image. "
        "Use null if the value is not present.\n" + desc +
        '\nReturn JSON: {"boxes": {"<field>": [ymin, xmin, ymax, xmax] | null}}'
    )


def zoom_prompt(doc_type, fields):
    desc = "\n".join(f'- "{k}": {DOC_TYPES[doc_type]["fields"][k]["desc"]}' for k in fields)
    return (
        f"Document type: {DOC_TYPES[doc_type]['label']}.\n"
        "You are given zoomed-in crops, one per field, in the order listed. Each crop shows the handwritten value "
        "for that field. Transcribe each exactly, character by character, without correcting it. "
        "Ambiguous characters: give your best guess and list alternative full values.\n" + desc +
        '\nReturn JSON: {"fields": {"<field>": {"value": <string|null>, "confidence": <0-1>, "alternatives": [<strings>]}}}'
    )


# ------------------------------------------------------------------ LLM stages
def classify(llm, im, tag):
    from common.llm import image_part

    out = llm.json([image_part(to_jpeg(im)), classify_prompt()], system=SYSTEM, temperature=0.0, tag=("classify", tag))
    dt = out.get("doc_type", "unknown")
    if dt not in DOC_TYPES:
        dt = "unknown"
    return dt, float(out.get("confidence", 0) or 0), out.get("reason", "")


def extract_once(llm, im, doc_type, tag, i, temperature):
    from common.llm import image_part

    out = llm.json([image_part(to_jpeg(im)), extract_prompt(doc_type)], system=SYSTEM,
                   temperature=temperature, tag=("extract", tag, i))
    return out.get("fields", {}) or {}


def zoom_read(llm, im_full, doc_type, tag):
    """Locate each hard field, crop + upscale it, and re-read all crops in one call."""
    from common.llm import image_part

    hard = [k for k, f in DOC_TYPES[doc_type]["fields"].items() if f["hard"]]
    if not hard:
        return {}
    boxes = (llm.json([image_part(to_jpeg(im_full)), locate_prompt(doc_type, hard)], system=SYSTEM,
                      temperature=0.0, tag=("locate", tag)) or {}).get("boxes", {}) or {}
    names, parts = [], []
    for k in hard:
        b = boxes.get(k)
        if not b or len(b) != 4:
            continue
        try:
            c = crop_box(im_full, b)
        except (TypeError, ValueError):
            c = None
        if c is None:
            continue
        names.append(k)
        parts.append(image_part(to_jpeg(c)))
    if not names:
        return {}
    result = {}
    for i in range(0, len(names), 4):  # <=4 images per request keeps us inside every provider's image limit
        out = llm.json(parts[i:i + 4] + [zoom_prompt(doc_type, names[i:i + 4])], system=SYSTEM, temperature=0.0,
                       tag=("zoom", tag, i // 4))
        result.update((out or {}).get("fields", {}) or {})
    return result


# ------------------------------------------------------------------ Tesseract cross-check (printed docs)
def tesseract_text(im):
    try:
        import pytesseract

        return pytesseract.image_to_string(im)
    except Exception:  # noqa: BLE001  (binary missing, etc.)
        return None


def tess_supports(kind, value, text):
    if text is None or not value:
        return None
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    if kind == "date":
        d = V.parse_date(value)
        if not d:
            return None
        return d.strftime("%d%m%Y") in t or d.strftime("%d%m%y") in t
    if kind in ("name", "address", "text", "place", "bank"):
        toks = [w for w in re.sub(r"[^A-Z0-9 ]", " ", str(value).upper()).split() if len(w) > 2 and w not in ("MR", "MRS")]
        return bool(toks) and all(w in text.upper() for w in toks)
    key = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return key in t if key else None


# ------------------------------------------------------------------ scoring
def _vote(kind, field, raw):
    """-> (key, cleaned_value) after validator-driven repair so 'SBINO..' and 'SBIN0..' agree."""
    if raw in (None, "", "null"):
        return None, None
    st, _, fixed = V.validate(kind, str(raw), field)
    cand = fixed if st != "invalid" else str(raw).strip()
    return V.normalize(kind, cand), cand


def score_field(field, fspec, votes, tess_ok=None):
    """votes: list of dicts {value, confidence, weight, source, alternatives}"""
    kind = fspec["kind"]
    tally = defaultdict(float)
    members = defaultdict(list)
    for v in votes:
        key, cleaned = _vote(kind, field, v.get("value"))
        w = v.get("weight", 1.0)
        tally[key] += w
        members[key].append((w, cleaned, v))
    total = sum(tally.values()) or 1.0
    modal = max(tally, key=lambda k: (tally[k], k is not None))
    if modal is None:
        return dict(value=None, normalized=None, confidence=0.0, components=dict(model_conf=0, agreement=tally[None] / total, validator="na"),
                    validator=dict(status="na", notes=[]), flags=["no value extracted (blank or illegible)"],
                    votes=[v.get("value") for v in votes], alternatives=[])
    mem = members[modal]
    best = max(mem, key=lambda m: (m[2].get("source") == "zoom", m[0]))
    value = best[1]
    model_conf = sum(m[0] * float(m[2].get("confidence") or 0.5) for m in mem) / sum(m[0] for m in mem)
    agreement = tally[modal] / total
    if tess_ok is not None:  # printed documents: Tesseract acts as the second voter
        agreement = 1.0 if tess_ok else 0.8
    st, notes, fixed = V.validate(kind, value, field)
    if st != "invalid" and fixed:
        value = fixed
    base = 0.5 * model_conf + 0.5 * agreement
    final = max(0.0, min(1.0, base * VALIDATOR_FACTOR[st]))
    flags = []
    if st in ("warn", "invalid"):
        flags += notes
    if agreement < 1.0 and tess_ok is None:
        others = sorted({str(v.get("value")) for v in votes if _vote(kind, field, v.get("value"))[0] != modal})
        flags.append(f"reads disagree ({agreement:.0%} agreement); other reads: {others}")
    if tess_ok is False:
        flags.append("Tesseract OCR could not corroborate this value")
    alts = sorted({a for _, _, v in mem for a in (v.get("alternatives") or []) if a and str(a) != str(value)})
    return dict(value=value, normalized=modal, confidence=round(final, 3),
                components=dict(model_conf=round(model_conf, 3), agreement=round(agreement, 3), validator=st),
                validator=dict(status=st, notes=notes), flags=flags, votes=[v.get("value") for v in votes], alternatives=alts)


def process_document(llm, path, tag, n_samples=3, use_zoom=True, use_tesseract=True):
    im0 = load_image(path)
    doc_type, cls_conf, reason = classify(llm, im0, tag)
    result = dict(file=path.name, document_type=dict(value=doc_type, confidence=round(cls_conf, 3), reason=reason))
    if doc_type == "unknown":
        result.update(handwritten=None, fields={}, flags=["document type not recognised"])
        return result
    spec = DOC_TYPES[doc_type]
    hw = spec["handwritten"]
    im = load_image(path, enhance=hw)
    per_field = {f: [] for f in spec["fields"]}

    n = n_samples if hw else 1
    for i in range(n):
        got = extract_once(llm, im, doc_type, tag, i, temperature=0.0 if i == 0 else 0.5)
        for f in spec["fields"]:
            g = got.get(f) or {}
            per_field[f].append(dict(value=g.get("value"), confidence=g.get("confidence"), weight=1.0, source="full",
                                     alternatives=g.get("alternatives")))
    if hw and use_zoom:
        try:
            zoom = zoom_read(llm, im, doc_type, tag)
        except Exception as e:  # noqa: BLE001 - zoom is a bonus pass; never lose the document over it
            print(f"  [warn] zoom pass failed for {path.name}: {e}")
            zoom = {}
        for f, g in zoom.items():
            if f in per_field and isinstance(g, dict):
                per_field[f].append(dict(value=g.get("value"), confidence=g.get("confidence"), weight=ZOOM_WEIGHT,
                                         source="zoom", alternatives=g.get("alternatives")))
    tess = tesseract_text(im0) if (not hw and use_tesseract) else None

    fields = {}
    for f, fspec in spec["fields"].items():
        votes = per_field[f]
        tess_ok = None
        if tess is not None:
            lead = next((v["value"] for v in votes if v["value"]), None)
            tess_ok = tess_supports(fspec["kind"], lead, tess)
        fields[f] = score_field(f, fspec, votes, tess_ok)
    result.update(handwritten=hw, fields=fields)
    return result
