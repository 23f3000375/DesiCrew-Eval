"""End-to-end tests over real HTTP against local mock providers (no internet, no API keys).
They test OUR code: client, fallback chain, tool-calling loop, and Q1/Q2/Q3 flows - not real model quality."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "q3_doc_pipeline", ROOT / "q2_doc_assistant"):
    sys.path.insert(0, str(p))

from tests.mock_server import MockProvider  # noqa: E402

GEM_MODELS = [{"id": "models/gemini-3-flash-preview"}, {"id": "models/gemini-2.5-flash-lite"},
              {"id": "models/gemini-2.5-flash"}, {"id": "models/gemini-2.5-flash-image"}, {"id": "models/gemini-2.5-pro"}]


@pytest.fixture
def world(monkeypatch):
    """gemini + groq mock servers wired in through *_BASE_URL; returns (gemini, groq, make_llm)."""
    gem = MockProvider("gemini", models=GEM_MODELS)
    groq = MockProvider("groq", models=[{"id": "meta-llama/llama-4-scout-17b-16e-instruct"}, {"id": "llama-3.3-70b-versatile"},
                                         {"id": "whisper-large-v3"}])
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY", "OPENAI_API_KEY", "OLLAMA_MODEL",
              "GEMINI_MODELS", "GROQ_MODELS", "LLM_PROVIDERS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-g")
    monkeypatch.setenv("GROQ_API_KEY", "test-q")
    monkeypatch.setenv("GEMINI_BASE_URL", gem.url)
    monkeypatch.setenv("GROQ_BASE_URL", groq.url)
    monkeypatch.setenv("LLM_MIN_INTERVAL", "0")
    monkeypatch.setenv("LLM_VERBOSE", "0")
    from common.llm import LLM
    yield gem, groq, LLM
    gem.stop()
    groq.stop()


def test_discovery_and_ordering(world):
    gem, groq, LLM = world
    labels = [c.label for c in LLM().candidates()]
    assert labels[0] == "gemini:gemini-2.5-flash"                 # stable, non-lite first
    assert "gemini:gemini-2.5-flash-image" not in labels and "gemini:gemini-2.5-pro" not in labels
    assert labels.index("gemini:gemini-3-flash-preview") > labels.index("gemini:gemini-2.5-flash-lite")  # previews last
    assert not any("whisper" in l for l in labels)
    assert any(l.startswith("groq:meta-llama/llama-4-scout") for l in labels)


def test_overload_falls_back_and_primary_cools_down(world):
    gem, groq, LLM = world
    gem.mode = "503"
    llm = LLM()
    assert llm.json("Reply with exactly this JSON: {\"ok\": true}") == {"ok": True}
    assert llm.last_used.provider == "groq"
    n503 = len(gem.requests)
    assert n503 >= 1
    llm.json("Reply with exactly this JSON: {\"ok\": true}")
    assert len(gem.requests) == n503, "gemini models should be on cooldown, not hammered again"


def test_json_mode_unsupported_is_switched_off(world):
    gem, groq, LLM = world
    gem.mode = "no_json"
    llm = LLM()
    assert llm.json("Reply with exactly this JSON: {\"ok\": true}") == {"ok": True}
    assert llm.last_used.provider == "gemini" and llm.last_used.json_mode is False
    assert len(gem.requests) == 2 and "response_format" not in gem.requests[1]


def test_vision_unsupported_skips_to_vision_model(world):
    gem, groq, LLM = world
    gem.mode = "no_vision"
    from common.llm import image_part
    from tests.test_integration_http import _png
    llm = LLM()
    out = llm.json([image_part(_png(), "image/png"), "Read the text in the image. Return JSON"])
    assert "4821" in out["text"] and llm.last_used.provider == "groq"
    before = len(gem.requests)
    llm.json([image_part(_png(), "image/png"), "Read the text in the image. Return JSON"])
    assert len(gem.requests) == before, "text-only flag must stick: no more image requests to that model"


def _png():
    import io
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(b, "PNG")
    return b.getvalue()


def test_bad_key_disables_provider_only(world):
    gem, groq, LLM = world
    gem.mode = "auth"
    llm = LLM()
    llm.json("Reply with exactly this JSON: {\"ok\": true}")
    llm.json("Reply with exactly this JSON: {\"ok\": true}")
    assert len(gem.requests) == 1 and llm.last_used.provider == "groq"


def test_everything_down_gives_clear_error(world, monkeypatch):
    gem, groq, LLM = world
    gem.mode = groq.mode = "503"
    monkeypatch.setenv("LLM_TOTAL_TIMEOUT", "0.01")
    with pytest.raises(RuntimeError) as e:
        LLM().json("Reply with exactly this JSON: {\"ok\": true}")
    assert "503" in str(e.value) or "busy" in str(e.value)


# ----------------------------------------------------------------------------- Q3 end to end
def test_q3_pipeline_over_http_with_flaky_primary(world, tmp_path, monkeypatch):
    gem, groq, LLM = world
    gem.mode = "alternate"                       # every second request to the primary is a 503
    import run_pipeline
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--out", str(out), "--data", str(ROOT / "q3_doc_pipeline" / "data")])
    run_pipeline.main()
    results = json.loads((out / "all_extractions.json").read_text(encoding="utf-8"))
    assert len(results) == 10 and all(r["document_type"]["value"] != "unknown" for r in results)
    import evaluate as E
    rows = E.compare(results, E.load_key(ROOT / "q3_doc_pipeline" / "answer_key.json"))
    assert all(r["correct"] for r in rows), [r for r in rows if not r["correct"]]
    by = {r["file"]: r for r in results}
    assert by["ECS.jpeg"]["fields"]["ifsc_code"]["value"] == "SBIN0227112"      # O/0 misread repaired
    flag = json.loads((out / "flag_report.json").read_text(encoding="utf-8"))
    assert {"pan_vs_tin", "fathers_name"} <= {f["group"] for f in flag["cross_document_findings"]}
    assert all(r["models_used"] for r in results)
    assert len(gem.requests) > 0 and len(groq.requests) > 0, "both providers should have served requests"
    # second run is free: everything cached
    calls_before = len(gem.requests) + len(groq.requests)
    run_pipeline.main()
    assert len(gem.requests) + len(groq.requests) == calls_before


# ----------------------------------------------------------------------------- Q1 end to end
def test_q1_agent_tool_loop_and_mid_conversation_failover(world):
    gem, groq, LLM = world
    from q1_inventory_agent.agent import InventoryAgent
    searches = []
    agent = InventoryAgent(LLM(), searcher=lambda q: (searches.append(q) or "Inventory turnover = COGS / average inventory.", ["Wiki - http://x"], None))

    r1 = agent.ask("Do the hand-in-stock numbers reconcile with opening + purchased - sold?")
    assert [s.kind for s in r1.steps] == ["code"] and r1.steps[0].error is None
    assert r1.steps[0].output.startswith("12"), r1.steps[0].output          # 12 of 46 do not reconcile (real sandbox + real data)
    assert "12" in r1.text

    gem.mode = "503"                                                       # primary dies mid-conversation
    r2 = agent.ask("What is inventory turnover? Look it up and calculate it.")
    assert [s.kind for s in r2.steps] == ["search", "code"] and searches == ["inventory turnover ratio formula"]
    assert agent.llm.last_used.provider == "groq"                          # history incl. tool calls accepted after the switch
    assert agent.history[0]["role"] == "user"

    r3 = agent.ask("Chart units sold for the top 10 products.")
    assert r3.charts and Path(r3.charts[0]).exists()


def test_search_fallbacks(monkeypatch):
    from q1_inventory_agent import search

    monkeypatch.setattr(search, "_ddg", lambda q, n: (_ for _ in ()).throw(RuntimeError("blocked")))
    monkeypatch.setattr(search, "_wikipedia", lambda q, n: [("Inventory turnover", "A ratio showing how many times inventory is sold.", "http://w/1")])
    text, sources, err = search.web_search("inventory turnover")
    assert "ratio" in text and sources == ["Inventory turnover - http://w/1"] and err is None
    monkeypatch.setattr(search, "_wikipedia", lambda q, n: (_ for _ in ()).throw(RuntimeError("offline")))
    text, sources, err = search.web_search("x")
    assert "unavailable" in text and sources == [] and err


def test_parse_args_tolerates_broken_json():
    from q1_inventory_agent.agent import parse_args
    assert parse_args('{"code": "print(1)"}') == {"code": "print(1)"}
    assert parse_args('{"code": "print(1)\nprint(2)"}')["code"] == "print(1)\nprint(2)"      # raw newline inside string
    assert parse_args("garbage") == {}


# ----------------------------------------------------------------------------- Q2 end to end
def test_q2_ingest_and_session_over_http(world, tmp_path):
    gem, groq, LLM = world
    import kb as K
    from session import SupportAssistant
    llm = LLM()
    rows = K.ingest(llm, "vision", out=tmp_path / "sections.json", docs_dir=ROOT / "q2_doc_assistant" / "docs")
    assert len(rows) >= 12                                                  # 4 pages x 4 mocked sections
    base = K.KnowledgeBase(rows)
    gem.mode = "503"                                                        # chat part runs on the fallback provider
    bot = SupportAssistant(base, llm)
    r1 = bot.ask("What is the assignment request form for?")
    assert r1["sources"] and "[1]" in r1["answer"]
    r2 = bot.ask("What is the claim settlement ratio?")
    assert r2["not_covered"] and not r2["sources"]
    assert llm.last_used.provider == "groq"


def test_doctor_probes_pass_against_mock(world):
    gem, groq, LLM = world
    import common.doctor as D
    llm = LLM()
    c = llm.candidates()[0]
    assert [D.probe(llm, c, k)[0] for k in ("json", "vision", "tools")] == ["PASS", "PASS", "PASS"]
