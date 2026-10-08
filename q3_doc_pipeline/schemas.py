"""Document types and the fields to extract for each (straight from the task table)."""

# kind -> drives normalisation + validation. hard=True -> high-value handwritten target
# that gets the extra "locate + zoom" second pass.
F = lambda desc, kind="text", hard=False: dict(desc=desc, kind=kind, hard=hard)

CHECKBOX_REASONS = [
    "Financial Planning",
    "Different investment / protection needs",
    "Separate policy purchased for different beneficiary",
    "Others",
]
FREQUENCIES = ["Monthly", "Quarterly", "Half-yearly", "Yearly", "As & when presented"]

DOC_TYPES = {
    "aadhaar": dict(
        label="Aadhaar Card", handwritten=False,
        cues="UIDAI / Government of India, 12-digit number in 4-4-4 groups, QR code, Aadhaar logo",
        fields={
            "aadhaar_number": F("12-digit Aadhaar number (4-4-4 groups)", "aadhaar"),
            "full_name": F("Name of the card holder", "name"),
            "date_of_birth": F("Date of birth as printed", "date"),
            "address": F("Full address block", "address"),
        }),
    "pan": dict(
        label="PAN Card", handwritten=False,
        cues="Income Tax Department, Permanent Account Number Card, 10-character alphanumeric PAN",
        fields={
            "pan_number": F("10-character PAN (AAAAA9999A)", "pan"),
            "full_name": F("Card holder name", "name"),
            "fathers_name": F("Father's name as printed", "name"),
            "date_of_birth": F("Date of birth as printed", "date"),
        }),
    "driving_licence": dict(
        label="Driving Licence", handwritten=False,
        cues="Indian Union Driving Licence, DL No., classes of vehicles, RTO",
        fields={
            "dl_number": F("Driving licence number", "dl"),
            "name": F("Holder name", "name"),
            "date_of_issue": F("Date of issue (DOI)", "date"),
            "valid_till": F("Valid-till date", "date"),
        }),
    "passport": dict(
        label="Passport", handwritten=False,
        cues="Republic of India passport data page with two machine readable lines (MRZ) at the bottom",
        fields={
            "passport_number": F("Passport number", "passport"),
            "date_of_birth": F("Date of birth as printed", "date"),
            "date_of_expiry": F("Date of expiry as printed", "date"),
            "mrz_line_2": F("The SECOND (bottom) MRZ line, exactly as printed, all 44 characters including '<'", "mrz"),
        }),
    "nach_ecs": dict(
        label="NACH / ECS Mandate", handwritten=True,
        cues="NACH Mandate Instruction form, UMRN, bank account number boxes, IFSC, debit amount",
        fields={
            "bank_account_number": F("Bank account number written in the boxes (digits only)", "account", True),
            "ifsc_code": F("IFSC code written in the boxes (4 letters, 0, 6 alphanumerics)", "ifsc", True),
            "bank_name": F("Bank name written next to 'with bank'", "bank"),
            "amount_figures": F("Amount in FIGURES (the number next to the rupee sign)", "amount", True),
            "frequency": F("The ticked frequency option, one of: " + ", ".join(FREQUENCIES), "frequency"),
        }),
    "fatca": dict(
        label="FATCA Annexure Form", handwritten=True,
        cues="Annexure Form, information required under Section 285BA of the Income Tax Act, tax residency table",
        fields={
            "policy_number": F("Policy No written in Section 1 (digits)", "policy_no", True),
            "tin_or_pan": F("Tax Identification Number / functional equivalent from the tax residency table", "tin", True),
            "fathers_name": F("Father's Name in Section 3", "name"),
            "place_of_birth": F("Place of birth in Section 3", "place", True),
            "nationality": F("Nationality in Section 3", "text"),
        }),
    "benefit_illustration": dict(
        label="Benefit Illustration Declaration", handwritten=True,
        cues="Customer Declaration - Benefit Illustration, Application/Proposal Form Number",
        fields={
            "application_number": F("Application / Proposal Form Number (digits)", "policy_no", True),
            "policyholder_name": F("Name of Policyholder (written next to 'Name of Policyholder' near the signature)", "name"),
            "date": F("Handwritten date next to the signature", "date", True),
            "place": F("Handwritten place next to the signature", "place", True),
        }),
    "moral_hazard": dict(
        label="Moral Hazard Questionnaire", handwritten=True,
        cues="Moral Hazard Questionnaire, dependents table, nominee questions",
        fields={
            "application_number": F("Application No. (digits)", "policy_no", True),
            "name_of_life_assured": F("Name of the Life to be Assured", "name"),
            "nominee_relationship": F("Exact relationship of nominee (question 5)", "text"),
            "date": F("Handwritten date in the Declaration of Life to be Assured", "date", True),
            "place": F("Handwritten place in the Declaration of Life to be Assured", "place", True),
        }),
    "multiple_policies": dict(
        label="Multiple Policies Consent Form", handwritten=True,
        cues="Split & Multiple Policies - Customer Consent form, reason for buying multiple policies checkboxes",
        fields={
            "proposer_name": F("Proposer / Life Assured name", "name"),
            "reason_for_multiple_policies": F("The TICKED checkbox, exactly one of: " + " | ".join(CHECKBOX_REASONS), "reason"),
            "date": F("Handwritten date at the bottom", "date", True),
            "place": F("Handwritten place at the bottom", "place", True),
        }),
    "suitability_profiler": dict(
        label="Suitability Profiler Declaration", handwritten=True,
        cues="Customer Declaration - Suitability Profiler, Name of Agent/SP",
        fields={
            "application_number": F("Application/Proposal Form Number (digits)", "policy_no", True),
            "name_of_life_assured": F("Name of Life Assured", "name"),
            "name_of_agent_sp": F("Name of Agent/SP", "name"),
            "date": F("Handwritten date next to the Life Assured's signature box", "date", True),
            "place": F("Handwritten place next to the Life Assured's signature box", "place", True),
        }),
}
