"""Graceful degradation: LLM failures, bad PDFs, missing OA copies, failing sources, empty results."""
import httpx
import pytest

from app.core.config import Settings
from app.extraction.acquisition import Acquirer
from app.extraction.text import pdf_to_sections
from app.literature.http import PoliteClient
from app.literature.sources import LiteratureSource
from app.pipelines.analysis import AnalysisPipeline, PipelineError
from app.providers.llm import BudgetedLLM, LLMError, LLMProvider, build_llm
from app.providers.llm.base import parse_json_object
from app.schemas.api import AnalyzeRequest
from tests.conftest import make_paper


class FlakyLLM(LLMProvider):
    name = "flaky"

    def __init__(self):
        super().__init__("m")
        self.calls = 0

    def is_available(self):
        return True

    def complete_json(self, system, user, schema, max_tokens=400):
        self.calls += 1
        if self.calls % 2:
            raise LLMError("timeout")
        return {"ok": True}


def test_budgeted_llm_isolates_failures_and_enforces_budget():
    provider = FlakyLLM()
    llm = BudgetedLLM(provider, max_calls=3)
    results = [llm.complete_json("s", "u", {}) for _ in range(5)]
    assert results[:3] == [None, {"ok": True}, None]
    assert results[3:] == [None, None]  # budget exhausted → no further provider calls
    assert provider.calls == 3
    assert len(llm.failures) == 2


def test_parse_json_object_is_lenient_but_strict_on_garbage():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": 2} hope this helps') == {"a": 2}
    with pytest.raises(LLMError):
        parse_json_object("no json here")


def test_unreachable_llm_degrades_to_rules(tmp_path):
    s = Settings(llm_provider="ollama", ollama_base_url="http://127.0.0.1:9", llm_model="missing:1b", _env_file=None)
    llm, warning = build_llm(s, use_llm=None)
    assert not llm.enabled
    assert "not reachable" in warning


def test_misconfigured_api_provider_is_reported():
    s = Settings(llm_provider="openai_compatible", llm_api_base_url="", _env_file=None)
    llm, warning = build_llm(s, use_llm=True)
    assert not llm.enabled and "LLM_API_BASE_URL" in warning


def test_invalid_pdf_rejected():
    with pytest.raises(ValueError):
        pdf_to_sections(b"<html>not a pdf</html>")
    with pytest.raises(ValueError):
        pdf_to_sections(b"%PDF-1.4 garbage")


def test_acquisition_falls_back_to_abstract(tmp_path):
    def handler(req):
        return httpx.Response(200, content=b"<html>paywall</html>", headers={"Content-Type": "text/html"})

    client = PoliteClient(transport=httpx.MockTransport(handler))
    client.sleep = lambda s: None
    paper = make_paper("p", pdf_url="https://arxiv.org/pdf/0000.00000")
    doc = Acquirer(client, tmp_path).acquire(paper)
    assert doc.paper.text_source == "abstract"
    assert "not a PDF" in doc.paper.acquisition_note
    assert doc.sections[0].name == "abstract"


def test_acquisition_without_any_text(tmp_path):
    client = PoliteClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    client.sleep = lambda s: None
    doc = Acquirer(client, tmp_path).acquire(make_paper("p").model_copy(update={"abstract": None}))
    assert doc.paper.text_source == "none" and doc.sections == []


class ExplodingSource(LiteratureSource):
    name = "exploding"

    def search(self, query, limit):
        raise RuntimeError("boom")


class EmptySource(LiteratureSource):
    name = "empty"

    def search(self, query, limit):
        return []


def test_pipeline_reports_no_papers(settings, models):
    warnings = []
    pipe = AnalysisPipeline(settings, warn=lambda st, m: warnings.append(m), models=models, sources=[ExplodingSource(), EmptySource()])
    with pytest.raises(PipelineError, match="No papers"):
        pipe.run(AnalyzeRequest(question="Does anything work at all?", demo=False))
    assert any("exploding" in w for w in warnings)


def test_pipeline_survives_extraction_error_for_one_paper(settings, models, monkeypatch):
    from app.extraction import claims as claims_mod

    original = claims_mod.ClaimExtractor.extract
    state = {"n": 0}

    def sometimes_broken(self, question, qvec, doc):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("model crashed")
        return original(self, question, qvec, doc)

    monkeypatch.setattr(claims_mod.ClaimExtractor, "extract", sometimes_broken)
    warnings = []
    result = AnalysisPipeline(settings, warn=lambda st, m: warnings.append(m), models=models).run(
        AnalyzeRequest(question="Do vision transformers need large-scale pre-training?", demo=True))
    assert result.claims
    assert any("extraction failed" in w for w in warnings)
