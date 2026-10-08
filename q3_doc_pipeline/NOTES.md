# Q3 - Notes: printed vs handwritten handling, confidence, failure cases

## Pipeline
`classify` (vision LLM, 10 types) -> `extract` (type-specific prompt, JSON) -> `score` (per-field confidence) ->
`cross-check` (documents vs each other) -> `flag` (fields/documents below threshold) -> JSON + `flag_report.md`.
The filename is never shown to the model: classification is from the image alone.

## Printed vs handwritten
| | printed (Aadhaar, PAN, DL, passport) | handwritten (6 HDFC forms) |
|---|---|---|
| reads | 1 greedy LLM read | 3 full-page reads (1 greedy + 2 sampled at T=0.5) **+ 1 zoomed-crop read** |
| image | downscaled to <=2000px | same, plus mild contrast stretch (faint pen strokes) |
| prompt | field list | adds handwriting rules: transcribe char-by-char, never autocorrect, digits-vs-letters constraints per field (account no. = digits, IFSC = 4 letters + `0` + 6 alnum, PAN/TIN = 5L 4D 1L), list ambiguous alternatives, checkbox = ticked box only (struck-out options are not selected), ignore printed labels/placeholders |
| second opinion | Tesseract OCR corroborates the value (if installed) | vote agreement across the reads |
| high-value fields | - | **locate-then-zoom**: the model returns a bounding box for each hard field (IFSC, account no., TIN, policy/application no., dates, place), the crop is cut from the original image, upscaled to >=1200px, autocontrasted and re-read; that read gets weight 2 |
| validators | format + checksums | same; plus confusable-character repair (`O`<->`0`, `I`/`L`<->`1`, `S`<->`5`, `B`<->`8` chosen by the position's expected type) before votes are compared |

Which model reads what: calls go through a fallback chain (see the top-level README), so a document can be read by more than one model if a provider is overloaded; `models_used` in each extraction JSON records this. Mixed-model votes are fine for self-consistency, but weaker fallback models are less accurate on handwriting, which is exactly what the agreement score, validators and cross-document checks are there to expose.

Why not just Tesseract for handwriting: it cannot read the pen-written fields on these scans (I tried it on the Q2 scans; the output was unusable), so it is only a cheap cross-check on printed text.

## Per-field confidence
`final = (0.5 * model_confidence + 0.5 * agreement) * validator_factor`, then cross-document adjustments.
- model_confidence: the model's own estimate for the winning value (known to be over-confident on handwriting, hence only half the score).
- agreement: share of weighted votes (3 full reads + zoom x2) matching the winning value, after validator repair. Printed: 1.0 if Tesseract corroborates, 0.8 if not.
- validator_factor: valid/not-applicable 1.0, warning 0.93, failed 0.6.
- cross-document: agreement between >=2 documents adds +0.03 each (max +0.09, **only if the field's own reads were unanimous**); disagreement multiplies by 0.8 and forces review.

## Threshold: 0.80
See `out/flag_report.md` (generated, includes a sweep against the answer key). Short version: errs on the side of flagging because a wrong IFSC/account/TIN breaks NACH debits, while a human check is cheap. A field is also flagged regardless of score if it is missing, fails its validator, or conflicts with another document.

## Things confirmed in the sample data (independent of any model run)
These come from the validators/answer key, not from an LLM run:
- Passport MRZ line 2: passport-number check digit validates, but the **DOB and expiry check digits fail** (printed `5` and `0`; computed `2` and `9`). The pipeline reports the line as printed and flags it.
- PAN card number `ABCDE1234F`: valid structure but 4th character `D` is not a real holder-type code -> warning.
- Aadhaar `1234 5678 9012`: starts with `1` and fails the Verhoeff checksum -> warning (it is a sample card).
- **Cross-document conflicts** the pipeline is designed to surface: FATCA TIN `BPQPD3051R` != PAN card `ABCDE1234F`; father is `Arjun Das Kumar` on FATCA but `Kumar` on the PAN card (partial match).
- The FATCA TIN's `1` is easily misread as `I`/`l`; the PAN-format validator resolves it by position.

## Failure cases observed in your live run
_Fill this in after `python q3_doc_pipeline/run_pipeline.py` + `python q3_doc_pipeline/evaluate.py`. Take them from `out/flag_report.md` and the "Incorrect / missing fields" block of the evaluator. Useful things to note for each: document, field, what was read vs the truth, which signal (agreement, validator, cross-doc) did or did not catch it._

-
-

## Honest limitations
- The answer key (`answer_key.json`) was transcribed by Claude from the sample images, with the high-value handwritten fields zoom-checked, but it is a draft: verify it against the images before quoting accuracy numbers.
- Confidence is a heuristic, not a calibrated probability. With only 10 documents there is not enough data to calibrate it.
- Self-consistency voting only catches errors that are *unstable* across reads. A consistently wrong read (the same misread digit every time) passes the vote and is only caught if a validator or cross-document check disagrees.
- The unit tests (`pytest`) use a simulated LLM to test the scoring/repair/flagging logic. They do not measure real model accuracy.
