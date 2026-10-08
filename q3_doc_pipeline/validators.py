"""Format / checksum validators + normalisation. Pure python, no LLM, fully unit-tested.

Each validator returns (status, notes, fixed_value):
    status: "valid" | "warn" | "invalid" | "na"
"""
import re
from datetime import date, datetime

# ---------- Verhoeff (Aadhaar checksum) ----------
_D = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],
      [4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],
      [8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]]
_P = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],
      [9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]


def verhoeff_ok(num: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


# ---------- confusable-character repair ----------
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "l": "1", "|": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
_TO_ALPHA = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}


def _digits(s: str) -> str:
    return "".join(_TO_DIGIT.get(c, c) for c in s)


def _clean(s) -> str:
    return re.sub(r"[\s\-]", "", str(s or "")).upper()


# ---------- date handling ----------
def parse_date(s):
    """Return a date or None. Accepts dd/mm/yyyy, dd-mm-yy, yyyy-mm-dd, '26 04 2026'."""
    if not s:
        return None
    t = re.sub(r"[.\s,\\|]+", "/", str(s).strip())
    t = re.sub(r"[-]+", "/", t)
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d"):
        try:
            d = datetime.strptime(t, fmt).date()
            if fmt == "%d/%m/%y":  # 2-digit year: forms here are all 20xx, DOBs would be 19xx
                d = d.replace(year=d.year if d.year <= 2049 else d.year - 100)
            return d
        except ValueError:
            continue
    return None


def validate_date(v, field=""):
    d = parse_date(v)
    if d is None:
        return "invalid", ["not a parseable date"], v
    notes = []
    today = date.today()
    if field in ("date_of_birth",) and not (date(1900, 1, 1) < d < today):
        return "invalid", ["DOB outside plausible range"], v
    if field == "date" and (d > today or d.year < 2015):
        return "warn", [f"form date {d.isoformat()} looks implausible"], d.strftime("%d/%m/%Y")
    return "valid", notes, d.strftime("%d/%m/%Y")


# ---------- identifiers ----------
def validate_aadhaar(v):
    n = _digits(_clean(v))
    if not re.fullmatch(r"\d{12}", n):
        return "invalid", ["Aadhaar must be 12 digits"], v
    notes = []
    if n[0] in "01":
        notes.append("Aadhaar never starts with 0/1")
    if not verhoeff_ok(n):
        notes.append("Verhoeff checksum fails (dummy/sample number, or a misread digit)")
    return ("warn" if notes else "valid"), notes, n


PAN_ENTITY = set("ABCFGHLJPT")


def _repair_pan(s):
    s = list(s)
    if len(s) != 10:
        return "".join(s)
    for i in range(10):
        if i < 5 or i == 9:
            s[i] = _TO_ALPHA.get(s[i], s[i]) if s[i].isdigit() else s[i]
        else:
            s[i] = _TO_DIGIT.get(s[i], s[i]) if s[i].isalpha() else s[i]
    return "".join(s)


def validate_pan(v):
    raw = _clean(v)
    fixed = _repair_pan(raw)
    notes = []
    if not re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", fixed):
        return "invalid", ["PAN must look like AAAAA9999A"], v
    if fixed != raw:
        notes.append(f"repaired confusable characters {raw} -> {fixed}")
    if fixed[3] not in PAN_ENTITY:
        return "warn", notes + [f"4th char '{fixed[3]}' is not a valid holder-type code"], fixed
    return ("warn" if notes else "valid"), notes, fixed


def validate_ifsc(v):
    raw = _clean(v)
    fixed = raw
    notes = []
    if len(raw) == 11:
        # 5th char is always zero; letters O/Q/D are the usual misreads
        if raw[4] != "0" and raw[4] in "OQD":
            fixed = raw[:4] + "0" + raw[5:]
            notes.append("5th character repaired to 0")
    if not re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", fixed):
        return "invalid", ["IFSC must be 4 letters + 0 + 6 alphanumerics"], v
    return ("warn" if notes else "valid"), notes, fixed


BANK_CODES = {"SBIN": "state bank of india", "HDFC": "hdfc bank", "ICIC": "icici bank", "UTIB": "axis bank",
              "PUNB": "punjab national bank", "BARB": "bank of baroda", "CNRB": "canara bank",
              "UBIN": "union bank of india", "KKBK": "kotak mahindra bank", "IDIB": "indian bank",
              "IOBA": "indian overseas bank", "BKID": "bank of india", "MAHB": "bank of maharashtra"}


def ifsc_bank_consistent(ifsc, bank_name):
    exp = BANK_CODES.get((ifsc or "")[:4])
    if not exp or not bank_name:
        return None
    b = re.sub(r"[^a-z ]", "", bank_name.lower())
    return exp.split()[0] in b and (exp.split()[-1] in b)


def validate_account(v):
    n = _digits(_clean(v))
    if not re.fullmatch(r"\d{9,18}", n):
        return "invalid", ["account number should be 9-18 digits"], v
    return "valid", [], n


def validate_policy_no(v):
    n = _digits(_clean(v))
    if not re.fullmatch(r"\d{8,16}", n):
        return "invalid", ["policy/application number should be 8-16 digits"], v
    return "valid", [], n


def validate_dl(v):
    raw = _clean(v)
    if not re.fullmatch(r"[A-Z]{2}\d{2}\d{4}\d{7}", raw):
        return "invalid", ["DL number should be SS99 YYYY NNNNNNN"], v
    yr = int(raw[4:8])
    if not 1950 <= yr <= date.today().year:
        return "warn", ["embedded issue year implausible"], raw
    return "valid", [], raw


def validate_passport(v):
    raw = _clean(v)
    if not re.fullmatch(r"[A-Z]\d{7}", raw):
        return "invalid", ["Indian passport number is 1 letter + 7 digits"], v
    return "valid", [], raw


def validate_tin(v):
    """Indian TIN on FATCA forms is normally the PAN."""
    raw = _clean(v)
    if re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", _repair_pan(raw)):
        return validate_pan(v)
    return "warn", ["TIN is not in PAN format - verify manually"], raw


def validate_amount(v):
    n = re.sub(r"[^\d.]", "", str(v or ""))
    if not n:
        return "invalid", ["no amount"], v
    return "valid", [], n.rstrip(".")


# ---------- MRZ (ICAO 9303 TD3 line 2) ----------
def _mrz_val(c):
    if c.isdigit():
        return int(c)
    if c.isalpha():
        return ord(c) - 55
    return 0


def mrz_check_digit(s: str) -> int:
    w = [7, 3, 1]
    return sum(_mrz_val(c) * w[i % 3] for i, c in enumerate(s)) % 10


def mrz_checks(line2: str) -> dict:
    """Return which check digits validate. line2 must be 44 chars."""
    if len(line2) != 44:
        return {"length_44": False}

    def ok(data, chk):
        return chk.isdigit() and mrz_check_digit(data) == int(chk)

    pn = line2[28:42]
    return {
        "length_44": True,
        "passport_no": ok(line2[0:9], line2[9]),
        "dob": ok(line2[13:19], line2[19]),
        "expiry": ok(line2[21:27], line2[27]),
        "personal_no": (line2[42] == "<") or ok(pn, line2[42]) or set(pn) == {"<"},
        "composite": ok(line2[0:10] + line2[13:20] + line2[21:43], line2[43]),
    }


def validate_mrz(v):
    raw = re.sub(r"\s", "", str(v or "")).upper()
    if len(raw) != 44:
        return "invalid", [f"MRZ line should be 44 chars, got {len(raw)}"], raw
    res = mrz_checks(raw)
    bad = [k for k, ok in res.items() if not ok]
    if bad:
        return "warn", [f"MRZ check digit(s) failed: {', '.join(bad)}"], raw
    return "valid", [], raw


# ---------- dispatcher ----------
def validate(kind, value, field=""):
    if value in (None, ""):
        return "na", ["no value"], value
    fn = {"aadhaar": validate_aadhaar, "pan": validate_pan, "ifsc": validate_ifsc, "account": validate_account,
          "policy_no": validate_policy_no, "dl": validate_dl, "passport": validate_passport, "tin": validate_tin,
          "amount": validate_amount, "mrz": validate_mrz}.get(kind)
    if kind == "date":
        return validate_date(value, field)
    if fn is None:
        return "na", [], value
    return fn(value)


# ---------- normalisation for comparison (answer key / cross-doc) ----------
_TITLES = r"\b(mr|mrs|ms|miss|shri|smt|s/o|d/o|w/o|c/o|son of|daughter of)\b\.?"


def normalize(kind, value):
    if value in (None, ""):
        return None
    s = str(value).strip()
    if kind == "date":
        d = parse_date(s)
        return d.isoformat() if d else re.sub(r"\W", "", s).lower()
    if kind in ("aadhaar", "account", "policy_no"):
        return _digits(_clean(s))
    if kind in ("pan", "tin"):
        return _repair_pan(_clean(s))
    if kind in ("ifsc", "dl", "passport"):
        return _clean(s)
    if kind == "mrz":
        return re.sub(r"<+", "<", re.sub(r"\s", "", s.upper()))
    if kind == "amount":
        return re.sub(r"[^\d]", "", s.split(".")[0])
    if kind in ("name", "place", "text", "bank", "reason", "address", "frequency"):
        t = re.sub(_TITLES, " ", s.lower())
        t = re.sub(r"[^a-z0-9& ]", " ", t)
        return re.sub(r"\s+", " ", t).strip()
    return s.lower()
