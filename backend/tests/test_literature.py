"""Literature retrieval: source adapters, rate-limit handling, deduplication, ranking."""
import httpx
import pytest

from app.literature.discovery import deduplicate, discover, rank_papers
from app.literature.http import PoliteClient, RateLimited, SourceError
from app.literature.sources import ArxivSource, LiteratureSource, OpenAlexSource, SemanticScholarSource, reconstruct_abstract
from tests.conftest import make_paper

OPENALEX_PAYLOAD = {
    "results": [
        {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.48550/arXiv.2010.11929",
            "display_name": "An Image is Worth 16x16 Words",
            "publication_year": 2020,
            "authorships": [{"author": {"display_name": "Alexey Dosovitskiy"}}, {"author": {"display_name": "Lucas Beyer"}}],
            "primary_location": {"landing_page_url": "http://arxiv.org/abs/2010.11929", "source": {"display_name": "arXiv"}},
            "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/2010.11929", "landing_page_url": "http://arxiv.org/abs/2010.11929"},
            "open_access": {"is_oa": True, "oa_status": "green"},
            "abstract_inverted_index": {"Transformers": [0], "work": [1], "well": [2]},
            "cited_by_count": 40000,
        },
        {"id": "https://openalex.org/W2", "display_name": None},  # malformed: no title → skipped
    ]
}

ARXIV_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2012.12877v2</id>
    <published>2020-12-23T18:42:10Z</published>
    <title>Training data-efficient image
      transformers</title>
    <summary>We produce competitive convolution-free transformers.</summary>
    <author><name>Hugo Touvron</name></author>
    <author><name>Matthieu Cord</name></author>
  </entry>
</feed>"""


def client_with(handler) -> PoliteClient:
    c = PoliteClient(transport=httpx.MockTransport(handler))
    c.sleep = lambda s: None
    return c


def test_reconstruct_abstract():
    assert reconstruct_abstract({"world": [1], "hello": [0]}) == "hello world"
    assert reconstruct_abstract(None) is None


def test_openalex_parses_and_skips_malformed():
    client = client_with(lambda req: httpx.Response(200, json=OPENALEX_PAYLOAD))
    papers = OpenAlexSource(client).search("vision transformers", 10)
    assert len(papers) == 1
    p = papers[0]
    assert p.doi == "10.48550/arxiv.2010.11929"
    assert p.source_ids["arxiv"] == "2010.11929"
    assert p.pdf_url == "https://arxiv.org/pdf/2010.11929"
    assert p.abstract == "Transformers work well"
    assert p.authors[0] == "Alexey Dosovitskiy"


def test_openalex_sends_key_and_title_abstract_filter():
    seen = {}

    def handler(req):
        seen.update(dict(req.url.params))
        return httpx.Response(200, json={"results": []})

    OpenAlexSource(client_with(handler), api_key="k123", email="me@example.org").search("vit: cnn", 5)
    assert seen["api_key"] == "k123"
    assert seen["filter"].startswith("title_and_abstract.search:vit  cnn")


def test_arxiv_feed_parsing():
    papers = ArxivSource.parse_feed(ARXIV_FEED)
    assert papers[0].paper_id == "arxiv:2012.12877"
    assert papers[0].title == "Training data-efficient image transformers"
    assert papers[0].year == 2020
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2012.12877"


def test_arxiv_malformed_xml_raises_source_error():
    with pytest.raises(SourceError):
        ArxivSource.parse_feed(b"<feed><entry>")


def test_rate_limit_retries_then_raises():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "1"})

    client = client_with(handler)
    with pytest.raises(RateLimited):
        SemanticScholarSource(client).search("q", 5)
    assert calls["n"] == client.max_retries + 1


def test_retry_recovers_after_transient_503():
    responses = iter([httpx.Response(503), httpx.Response(200, json={"data": [{"paperId": "abc", "title": "A paper", "year": 2021}]})])
    papers = SemanticScholarSource(client_with(lambda req: next(responses))).search("q", 5)
    assert papers[0].paper_id == "s2:abc"


class BrokenSource(LiteratureSource):
    name = "broken"

    def search(self, query, limit):
        raise RateLimited("HTTP 429")


class StaticSource(LiteratureSource):
    name = "static"

    def __init__(self, papers):
        self.papers = papers

    def search(self, query, limit):
        return self.papers


def test_discover_degrades_when_a_source_fails():
    result = discover(["q"], [BrokenSource(), StaticSource([make_paper("a"), make_paper("b")])], 10)
    assert len(result.candidates) == 2
    assert any("broken" in w for w in result.warnings)


def test_deduplicate_by_doi_arxiv_and_title():
    a = make_paper("openalex:1", doi="10.1/x", sources=["openalex"])
    b = make_paper("s2:1", doi="https://doi.org/10.1/X", sources=["semantic_scholar"], citation_count=10)
    c = make_paper("arxiv:1", source_ids={"arxiv": "2101.00001"}, sources=["arxiv"], pdf_url="https://arxiv.org/pdf/2101.00001")
    d = make_paper("s2:2", source_ids={"arxiv": "2101.00001"}, sources=["semantic_scholar"])
    e = make_paper("x:1").model_copy(update={"title": "Deep Residual Learning for Image Recognition"})
    f = make_paper("x:2").model_copy(update={"title": "Deep residual learning for image recognition."})
    out = deduplicate([a, b, c, d, e, f])
    assert len(out) == 3
    merged = next(p for p in out if p.doi == "10.1/x")
    assert set(merged.sources) == {"openalex", "semantic_scholar"}
    assert merged.citation_count == 10


def test_ranking_prefers_relevant_papers(models):
    relevant = make_paper("r").model_copy(update={"title": "Vision transformers on small datasets", "abstract": "vision transformers small datasets pre-training"})
    other = make_paper("o").model_copy(update={"title": "Soil microbiome of wheat fields", "abstract": "soil bacteria wheat"})
    ranked = rank_papers("vision transformers small datasets", [other, relevant], models.reranker)
    assert ranked[0].paper_id == "r"
    assert ranked[0].relevance > ranked[1].relevance
