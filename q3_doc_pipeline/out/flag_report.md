# Human-review flag report

**Threshold: 0.80** (per-field confidence; configurable with `--threshold`).

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


## Threshold sweep vs draft answer key

| threshold | errors caught | correct fields flagged |
|---|---|---|
| 0.60 | 2/2 | 0/42 |
| 0.70 | 2/2 | 0/42 |
| 0.80 | 2/2 | 3/42 |
| 0.85 | 2/2 | 3/42 |
| 0.90 | 2/2 | 4/42 |

```
Field-level accuracy vs answer key
----------------------------------------
overall            42/44 (95%)
printed fields     16/16 (100%)
handwritten fields 26/28 (93%)
high-value handwritten (IFSC/TIN/account/date/place) 10/12 (83%)

Per document:
  Aadhar.png                                   4/4 (100%)
  ChatGPT Image May 2, 2026, 03_43_11 PM.png   4/4 (100%)
  ChatGPT Image May 2, 2026, 03_52_54 PM.png   4/4 (100%)
  ECS.jpeg                                     5/5 (100%)
  Fatca.jpeg                                   5/5 (100%)
  ID.png                                       4/4 (100%)
  Illustration.jpeg                            3/4 (75%)
  Moral.jpeg                                   4/5 (80%)
  split.jpeg                                   4/4 (100%)
  suitability.jpeg                             5/5 (100%)

Incorrect / missing fields:
  Illustration.jpeg              date                     got='28/04/2026' truth='2026-04-26' conf=0.5
  Moral.jpeg                     date                     got='06/04/2026' truth='2026-04-26' conf=0.586
```

## Documents

| file | type | class. conf | fields flagged | review? |
|---|---|---|---|---|
| Aadhar.png | aadhaar | 1.00 | 0/4 | no |
| ChatGPT Image May 2, 2026, 03_43_11 PM.png | driving_licence | 1.00 | 0/4 | no |
| ChatGPT Image May 2, 2026, 03_52_54 PM.png | passport | 1.00 | 0/4 | no |
| ECS.jpeg | nach_ecs | 1.00 | 0/5 | no |
| Fatca.jpeg | fatca | 1.00 | 2/5 | YES |
| ID.png | pan | 1.00 | 2/4 | YES |
| Illustration.jpeg | benefit_illustration | 1.00 | 1/4 | YES |
| Moral.jpeg | moral_hazard | 1.00 | 1/5 | YES |
| split.jpeg | multiple_policies | 1.00 | 1/4 | YES |
| suitability.jpeg | suitability_profiler | 0.99 | 0/5 | no |

## Flagged fields (7)

| document | field | value | conf | reasons |
|---|---|---|---|---|
| fatca | tin_or_pan | `BPQPD3051R` | 0.79 | low_confidence, cross_document_conflict: PAN on PAN card (ABCDE1234F) differs from TIN on FATCA form (BPQPD3051R) |
| fatca | fathers_name | `Arjun Das Kumar` | 0.90 | cross_document_conflict: father's name partial match: PAN 'S/O KUMAR' vs FATCA 'Arjun Das Kumar' (one is a subset of the other) |
| pan | pan_number | `ABCDE1234F` | 0.74 | low_confidence, cross_document_conflict: 4th char 'D' is not a valid holder-type code; PAN on PAN card (ABCDE1234F) differs from TIN on FATCA form (BPQPD3051R) |
| pan | fathers_name | `S/O KUMAR` | 0.90 | cross_document_conflict: father's name partial match: PAN 'S/O KUMAR' vs FATCA 'Arjun Das Kumar' (one is a subset of the other) |
| benefit_illustration | date | `28/04/2026` | 0.50 | low_confidence, cross_document_conflict: reads disagree (40% agreement); other reads: ['20/04/2026', '22/04/2024', '29/04/2024']; form_date: differs from the majority value across documents (2/4 docs read '26/04/2026') |
| moral_hazard | date | `06/04/2026` | 0.59 | low_confidence, cross_document_conflict: reads disagree (60% agreement); other reads: ['26/04/2026']; form_date: differs from the majority value across documents (2/4 docs read '26/04/2026') |
| multiple_policies | date | `26/04/2026` | 0.77 | low_confidence: reads disagree (60% agreement); other reads: ['26/04/2026.'] |

## Cross-document findings

- **conflict** (form_date): benefit_illustration.date differs from majority {'benefit_illustration': '28/04/2026'}
- **conflict** (form_date): moral_hazard.date differs from majority {'moral_hazard': '06/04/2026'}
- **conflict** (pan_vs_tin): PAN on PAN card (ABCDE1234F) differs from TIN on FATCA form (BPQPD3051R) 
- **partial** (fathers_name): father's name partial match: PAN 'S/O KUMAR' vs FATCA 'Arjun Das Kumar' (one is a subset of the other) 