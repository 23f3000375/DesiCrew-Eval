"""Cross-document consistency. Same person/application should agree across documents.

* Agreement between >=2 docs  -> small confidence boost (independent corroboration)
* Disagreement                -> confidence x0.8 and a human-review flag
* PAN card number vs FATCA TIN, father's name partial matches get dedicated rules
"""
from collections import Counter, defaultdict

BOOST = 0.03
CONFLICT_FACTOR = 0.8

GROUPS = {
    "person_name": [("aadhaar", "full_name"), ("pan", "full_name"), ("driving_licence", "name"),
                    ("benefit_illustration", "policyholder_name"), ("moral_hazard", "name_of_life_assured"),
                    ("multiple_policies", "proposer_name"), ("suitability_profiler", "name_of_life_assured")],
    "date_of_birth": [("aadhaar", "date_of_birth"), ("pan", "date_of_birth"), ("passport", "date_of_birth")],
    "application_number": [("fatca", "policy_number"), ("benefit_illustration", "application_number"),
                           ("moral_hazard", "application_number"), ("suitability_profiler", "application_number")],
    "form_date": [("benefit_illustration", "date"), ("moral_hazard", "date"),
                  ("multiple_policies", "date"), ("suitability_profiler", "date")],
    "form_place": [("benefit_illustration", "place"), ("moral_hazard", "place"),
                   ("multiple_policies", "place"), ("suitability_profiler", "place")],
}


def _tokens(s):
    return set((s or "").split())


def run(docs):
    """docs: {doc_type: result_dict}. Mutates field confidences/flags; returns list of findings."""
    findings = []

    def fld(dt, f):
        d = docs.get(dt)
        return (d or {}).get("fields", {}).get(f)

    def penalise(dt, f, msg):
        x = fld(dt, f)
        if x:
            x["confidence"] = round(x["confidence"] * CONFLICT_FACTOR, 3)
            x["flags"].append(msg)
            x["cross_doc_conflict"] = True

    def boost(dt, f, n):
        x = fld(dt, f)
        # only corroborate fields whose own reads were unanimous; otherwise the boost would hide
        # within-document disagreement
        if x and x["value"] and x["components"]["agreement"] >= 0.999:
            x["confidence"] = round(min(1.0, x["confidence"] + BOOST * min(n, 3)), 3)
            x.setdefault("notes", []).append(f"corroborated by {n} other document(s)")

    for gname, members in GROUPS.items():
        present = [(dt, f, fld(dt, f)) for dt, f in members if fld(dt, f) and fld(dt, f)["normalized"]]
        if len(present) < 2:
            continue
        cnt = Counter(x["normalized"] for _, _, x in present)
        maj, maj_n = cnt.most_common(1)[0]
        if maj_n < 2:  # no majority at all: everyone disagrees
            for dt, f, x in present:
                penalise(dt, f, f"{gname}: no two documents agree ({[y['value'] for _, _, y in present]})")
            findings.append(dict(group=gname, severity="conflict", detail="no majority", values={dt: x["value"] for dt, _, x in present}))
            continue
        for dt, f, x in present:
            if x["normalized"] == maj:
                boost(dt, f, maj_n - 1)
            else:
                penalise(dt, f, f"{gname}: differs from the majority value across documents "
                                f"({maj_n}/{len(present)} docs read '{[y['value'] for _, _, y in present if y['normalized'] == maj][0]}')")
                findings.append(dict(group=gname, severity="conflict", detail=f"{dt}.{f} differs from majority",
                                     values={dt: x["value"]}))

    # PAN card number vs TIN on FATCA
    pan, tin = fld("pan", "pan_number"), fld("fatca", "tin_or_pan")
    if pan and tin and pan["normalized"] and tin["normalized"] and pan["normalized"] != tin["normalized"]:
        msg = f"PAN on PAN card ({pan['value']}) differs from TIN on FATCA form ({tin['value']})"
        penalise("pan", "pan_number", msg)
        penalise("fatca", "tin_or_pan", msg)
        findings.append(dict(group="pan_vs_tin", severity="conflict", detail=msg))

    # Father's name: PAN vs FATCA (and passport has none)
    a, b = fld("pan", "fathers_name"), fld("fatca", "fathers_name")
    if a and b and a["normalized"] and b["normalized"] and a["normalized"] != b["normalized"]:
        ta, tb = _tokens(a["normalized"]), _tokens(b["normalized"])
        if ta <= tb or tb <= ta:
            msg = f"father's name partial match: PAN '{a['value']}' vs FATCA '{b['value']}' (one is a subset of the other)"
            for dt, x in (("pan", a), ("fatca", b)):
                x["confidence"] = round(x["confidence"] * 0.9, 3)
                x["flags"].append(msg)
                x["cross_doc_conflict"] = True
            findings.append(dict(group="fathers_name", severity="partial", detail=msg))
        else:
            msg = f"father's name differs: PAN '{a['value']}' vs FATCA '{b['value']}'"
            penalise("pan", "fathers_name", msg)
            penalise("fatca", "fathers_name", msg)
            findings.append(dict(group="fathers_name", severity="conflict", detail=msg))

    # IFSC prefix vs bank name
    import validators as V
    ifsc, bank = fld("nach_ecs", "ifsc_code"), fld("nach_ecs", "bank_name")
    if ifsc and bank and ifsc["value"] and bank["value"]:
        ok = V.ifsc_bank_consistent(ifsc["value"], bank["value"])
        if ok is False:
            msg = f"IFSC prefix {ifsc['value'][:4]} does not match bank name '{bank['value']}'"
            penalise("nach_ecs", "ifsc_code", msg)
            penalise("nach_ecs", "bank_name", msg)
            findings.append(dict(group="ifsc_vs_bank", severity="conflict", detail=msg))
        elif ok is True:
            boost("nach_ecs", "ifsc_code", 1)
            boost("nach_ecs", "bank_name", 1)

    # Passport printed fields vs its own MRZ line 2
    pp = docs.get("passport")
    if pp:
        mrz = (fld("passport", "mrz_line_2") or {}).get("value") or ""
        if len(mrz) == 44:
            pn = fld("passport", "passport_number")
            if pn and pn["value"] and mrz[:9].replace("<", "") != pn["value"]:
                penalise("passport", "passport_number", "printed passport number differs from MRZ")
                penalise("passport", "mrz_line_2", "MRZ passport number differs from printed number")
                findings.append(dict(group="passport_mrz", severity="conflict", detail="passport number vs MRZ"))
            for f, sl in (("date_of_birth", slice(13, 19)), ("date_of_expiry", slice(21, 27))):
                x = fld("passport", f)
                if x and x["normalized"]:
                    d = x["normalized"].replace("-", "")[2:]  # YYYY-MM-DD -> YYMMDD
                    if d and d != mrz[sl]:
                        penalise("passport", f, f"printed {f} differs from MRZ ({mrz[sl]})")
                        findings.append(dict(group="passport_mrz", severity="conflict", detail=f))
    return findings
