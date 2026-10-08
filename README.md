# DesiCrew Solutions - Data Science Internship, first-round evaluation

Three solutions in one repo. They talk to LLMs through **one fallback-capable client** (`common/llm.py`), so an overloaded
model (the `503 ... high demand` error) no longer stops anything: the next model or provider is tried automatically.

| | folder | what it is |
|---|---|---|
| Q1 | `q1_inventory_agent/` | Chat agent over the inventory Excel: writes + runs pandas code (sandboxed subprocess), searches the web for definitions (DuckDuckGo, Wikipedia backup), explains in plain English. Streamlit UI. |
| Q2 | `q2_doc_assistant/` | Document-aware support assistant over the two scanned HDFC Life PDFs: vision-OCR ingestion, hybrid retrieval, section-level citations, session memory (no repeats), topic-switch handling, scripted 13-turn demo. Streamlit UI. |
| Q3 | `q3_doc_pipeline/` | 10-document classify -> extract -> score -> flag pipeline with a handwriting-specific path (multi-read voting + locate-and-zoom), validators/checksums, cross-document checks, evaluator + threshold sweep. |


## Setup (Windows, Git Bash - the commands that work there)
```bash
python -m venv .venv
source .venv/Scripts/activate          # Git Bash on Windows. (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python common/doctor.py                # shows which models work with YOUR keys (vision / tools / JSON)
```
Currently the env file has no API keys, After recording and before pushing the final code, I removed it from there.

## Run everything in one go
```bash
python run_all.py            # doctor -> Q3 pipeline -> Q3 evaluate -> Q1 demo -> Q2 KB + demo (keeps going if a step fails)
```
...or step by step:

### Q3 - document pipeline (most important)
```bash
python q3_doc_pipeline/run_pipeline.py           # ~44 LLM calls, roughly 5-10 min on free tiers; cached, so reruns are free
python q3_doc_pipeline/evaluate.py               # accuracy vs answer_key.json + threshold sweep
```
Outputs in `q3_doc_pipeline/out/`: `extractions/<doc>.json` (one per document, per-field confidence and `models_used`),
`all_extractions.json`, `flag_report.md` / `.json` (threshold + rationale + flagged fields).
Notes on handwriting and failure cases: `q3_doc_pipeline/NOTES.md`. Options: `--threshold 0.85`, `--fresh`, `--only ECS`, `--model groq:llama-3.3-70b-versatile`.

### Q1 - inventory agent
```bash
streamlit run q1_inventory_agent/app.py          # chat UI (press Enter if Streamlit asks for an email); sample questions in the sidebar
python q1_inventory_agent/run_demo.py            # optional: scripted run -> demo_transcript.md
```
*"What is inventory turnover? Look it up and calculate it."* (search + code), *"Chart units sold for the top 10 products."*
The code sandbox is a separate process with a timeout and restricted imports/builtins: demo-grade, not a security boundary.

### Q2 - document assistant
```bash
python q2_doc_assistant/kb.py                    # vision-OCR the scanned PDFs -> kb/sections.json   
streamlit run q2_doc_assistant/app.py            # chat UI
python q2_doc_assistant/run_demo.py              # scripted 13-turn conversation -> demo_transcript.md
```

## Troubleshooting
| symptom | cause / fix |
|---|---|
| `503 ... high demand` | Provider overload. Handled automatically; add a second provider key for real resilience. |
| `No usable model with vision left` | None of your providers' models can read images. Add `GEMINI_API_KEY` / `GROQ_API_KEY` / `OPENROUTER_API_KEY`; check `python common/doctor.py`. |
| `API key rejected` | Wrong/revoked key in `.env` (no quotes, no spaces). That provider is skipped, others continue. |
| A run died midway | Re-run the same command: finished Q3 calls are cached in `q3_doc_pipeline/out/raw_cache`. |
| Search says "unavailable" | DuckDuckGo blocked on your network; Wikipedia backup is tried; the agent tells the user it could not verify online. |
| `ModuleNotFoundError` | The venv is not active (`source .venv/Scripts/activate`) or `pip install -r requirements.txt` was interrupted: run it again. |

## Tests (no keys, no internet needed)
```bash
pytest -q          # 24 tests, ~1 min
```
* validators/checksums, Q3 scoring + cross-document logic, Q2 retrieval + session logic (simulated LLM)
* `tests/test_integration_http.py`: the real client against **local mock providers over HTTP** that return 503/429/401/unsupported-feature errors
  on demand: fallback and cooldown, model discovery order, JSON-mode / vision downgrade, bad-key handling, the full Q3 pipeline with a flaky primary
  (+ cache rerun), the Q1 tool-calling agent incl. switching provider mid-conversation, Q2 ingestion + session.
