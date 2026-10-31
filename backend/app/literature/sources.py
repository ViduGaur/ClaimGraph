"""Literature sources. Each adapter maps a public API onto :class:`Paper`.

| source            | key?      | free budget / limits (checked Sep 2026)                     |
|-------------------|-----------|-------------------------------------------------------------|
| OpenAlex          | optional  | $0.10/day keyless, $1/day with free key; search = $0.001    |
| Semantic Scholar  | optional  | shared keyless pool (often 429); free key = 1 req/s          |
| arXiv             | none      | ≤ 1 request every 3 s, single connection                    |
| Europe PMC        | none      | generous; biomedical literature                             |

Adapters never raise on bad records: malformed items are skipped and counted.
"""
from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path

from app.literature.http import PoliteClient, SourceError
from app.schemas.domain import Paper

log = logging.getLogger(__name__)


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _strip_markup(text: str | None) -> str:
    return _clean(re.sub(r"<[^>]+>", " ", text or ""))


def _pdf_like(url: str | None) -> str | None:
    """Only accept an OA URL as a PDF link when it plausibly is one."""
    if url and (url.lower().endswith(".pdf") or "arxiv.org/pdf" in url or "/pdf" in url):
        return url
    return None


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as an inverted index {word: [positions]}."""
    if not inverted:
        return None
    positions = [(pos, word) for word, poss in inverted.items() for pos in poss]
    return " ".join(w for _, w in sorted(positions)) or None


class LiteratureSource(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, limit: int) -> list[Paper]:
        """Return candidate papers. May raise SourceError."""


# --------------------------------------------------------------------------- OpenAlex
class OpenAlexSource(LiteratureSource):
    name = "openalex"
    BASE = "https://api.openalex.org/works"
    FIELDS = ("id,doi,display_name,publication_year,authorships,primary_location,open_access,"
              "best_oa_location,cited_by_count,abstract_inverted_index,ids,type,relevance_score")

    def __init__(self, client: PoliteClient, api_key: str = "", email: str = ""):
        self.client, self.api_key, self.email = client, api_key, email

    def search(self, query: str, limit: int) -> list[Paper]:
        term = re.sub(r"[,:|]", " ", query)
        params = {
            "filter": f"title_and_abstract.search:{term},type:article|preprint|review",
            "sort": "relevance_score:desc",
            "per_page": min(limit, 50),
            "select": self.FIELDS,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["mailto"] = self.email
        resp = self.client.get(self.BASE, params=params)
        if resp.status_code in (401, 403):
            raise SourceError("OpenAlex refused the request (daily keyless budget exhausted or invalid key)")
        if resp.status_code >= 400:
            raise SourceError(f"OpenAlex HTTP {resp.status_code}")
        papers = []
        for item in resp.json().get("results", []):
            try:
                papers.append(self._to_paper(item))
            except Exception as exc:  # noqa: BLE001 — malformed metadata
                log.debug("skipping malformed OpenAlex record: %s", exc)
        return papers

    @staticmethod
    def _to_paper(item: dict) -> Paper:
        oa = item.get("open_access") or {}
        best = item.get("best_oa_location") or {}
        primary = item.get("primary_location") or {}
        source = primary.get("source") or {}
        ids = item.get("ids") or {}
        openalex_id = (item.get("id") or "").rsplit("/", 1)[-1]
        arxiv = None
        for loc in (best, primary):
            m = re.search(r"arxiv\.org/(?:abs|pdf)/([\d.]+)", (loc.get("landing_page_url") or "") + " " + (loc.get("pdf_url") or ""))
            if m:
                arxiv = m.group(1)
        source_ids = {"openalex": openalex_id}
        if arxiv:
            source_ids["arxiv"] = arxiv
        if ids.get("pmid"):
            source_ids["pmid"] = str(ids["pmid"]).rsplit("/", 1)[-1]
        title = _strip_markup(item.get("display_name"))
        if not title:
            raise ValueError("missing title")
        return Paper(
            paper_id=f"openalex:{openalex_id}",
            title=title,
            authors=[_clean((a.get("author") or {}).get("display_name")) for a in item.get("authorships") or [] if a.get("author")][:25],
            year=item.get("publication_year"),
            venue=_clean(source.get("display_name")) or None,
            doi=item.get("doi"),
            url=primary.get("landing_page_url") or item.get("doi") or item.get("id"),
            pdf_url=best.get("pdf_url") or _pdf_like(oa.get("oa_url")),
            abstract=reconstruct_abstract(item.get("abstract_inverted_index")),
            is_open_access=bool(oa.get("is_oa")),
            oa_status=oa.get("oa_status"),
            citation_count=item.get("cited_by_count"),
            sources=["openalex"],
            source_ids=source_ids,
        )


# --------------------------------------------------------------------------- Semantic Scholar
class SemanticScholarSource(LiteratureSource):
    name = "semantic_scholar"
    BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,abstract,year,authors,venue,externalIds,openAccessPdf,isOpenAccess,citationCount,url"

    def __init__(self, client: PoliteClient, api_key: str = ""):
        self.client, self.api_key = client, api_key

    def search(self, query: str, limit: int) -> list[Paper]:
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        resp = self.client.get(self.BASE, params={"query": query, "limit": min(limit, 100), "fields": self.FIELDS}, headers=headers)
        if resp.status_code >= 400:
            raise SourceError(f"Semantic Scholar HTTP {resp.status_code}")
        papers = []
        for item in resp.json().get("data") or []:
            try:
                papers.append(self._to_paper(item))
            except Exception as exc:  # noqa: BLE001
                log.debug("skipping malformed S2 record: %s", exc)
        return papers

    @staticmethod
    def _to_paper(item: dict) -> Paper:
        ext = item.get("externalIds") or {}
        pdf = (item.get("openAccessPdf") or {}).get("url") or None
        title = _clean(item.get("title"))
        if not title or not item.get("paperId"):
            raise ValueError("missing title/id")
        source_ids = {"s2": item["paperId"]}
        if ext.get("ArXiv"):
            source_ids["arxiv"] = ext["ArXiv"]
        if ext.get("PubMed"):
            source_ids["pmid"] = str(ext["PubMed"])
        return Paper(
            paper_id=f"s2:{item['paperId']}",
            title=title,
            authors=[_clean(a.get("name")) for a in item.get("authors") or [] if a.get("name")][:25],
            year=item.get("year"),
            venue=_clean(item.get("venue")) or None,
            doi=ext.get("DOI") or (f"10.48550/arXiv.{ext['ArXiv']}" if ext.get("ArXiv") else None),
            url=item.get("url"),
            pdf_url=pdf or None,
            abstract=_clean(item.get("abstract")) or None,
            is_open_access=bool(item.get("isOpenAccess") or pdf),
            oa_status="open" if pdf else None,
            citation_count=item.get("citationCount"),
            sources=["semantic_scholar"],
            source_ids=source_ids,
        )


# --------------------------------------------------------------------------- arXiv
class ArxivSource(LiteratureSource):
    name = "arxiv"
    BASE = "https://export.arxiv.org/api/query"
    NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    def __init__(self, client: PoliteClient):
        self.client = client

    def search(self, query: str, limit: int) -> list[Paper]:
        words = [w for w in re.findall(r"[A-Za-z0-9\-]+", query) if len(w) > 2][:8]
        if not words:
            return []
        search_query = " AND ".join(f"all:{w}" for w in words)
        resp = self.client.get(self.BASE, params={"search_query": search_query, "start": 0,
                                                  "max_results": min(limit, 50), "sortBy": "relevance"})
        if resp.status_code >= 400:
            raise SourceError(f"arXiv HTTP {resp.status_code}")
        return self.parse_feed(resp.content)

    def fetch_ids(self, arxiv_ids: list[str]) -> list[Paper]:
        """Metadata for known arXiv ids (one request), in the order given."""
        if not arxiv_ids:
            return []
        resp = self.client.get(self.BASE, params={"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)})
        if resp.status_code >= 400:
            raise SourceError(f"arXiv HTTP {resp.status_code}")
        found = {p.source_ids["arxiv"]: p for p in self.parse_feed(resp.content)}
        return [found[i] for i in arxiv_ids if i in found]

    @classmethod
    def parse_feed(cls, content: bytes) -> list[Paper]:
        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            raise SourceError(f"arXiv returned malformed XML: {exc}") from exc
        papers = []
        for entry in root.findall("a:entry", cls.NS):
            try:
                abs_url = entry.findtext("a:id", default="", namespaces=cls.NS)
                arxiv_id = re.sub(r"v\d+$", "", abs_url.rsplit("/abs/", 1)[-1])
                title = _clean(entry.findtext("a:title", default="", namespaces=cls.NS))
                if not arxiv_id or not title:
                    continue
                journal = _clean(entry.findtext("arxiv:journal_ref", default="", namespaces=cls.NS))
                doi = _clean(entry.findtext("arxiv:doi", default="", namespaces=cls.NS)) or f"10.48550/arXiv.{arxiv_id}"
                papers.append(Paper(
                    paper_id=f"arxiv:{arxiv_id}",
                    title=title,
                    authors=[_clean(a.findtext("a:name", default="", namespaces=cls.NS)) for a in entry.findall("a:author", cls.NS)][:25],
                    year=int(entry.findtext("a:published", default="0", namespaces=cls.NS)[:4] or 0) or None,
                    venue=journal or "arXiv",
                    doi=doi,
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                    abstract=_clean(entry.findtext("a:summary", default="", namespaces=cls.NS)) or None,
                    is_open_access=True,
                    oa_status="green",
                    sources=["arxiv"],
                    source_ids={"arxiv": arxiv_id},
                ))
            except Exception as exc:  # noqa: BLE001
                log.debug("skipping malformed arXiv entry: %s", exc)
        return papers


# --------------------------------------------------------------------------- Europe PMC
class EuropePMCSource(LiteratureSource):
    name = "europepmc"
    BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self, client: PoliteClient):
        self.client = client

    def search(self, query: str, limit: int) -> list[Paper]:
        resp = self.client.get(self.BASE, params={"query": query, "format": "json", "resultType": "core",
                                                  "pageSize": min(limit, 50)})
        if resp.status_code >= 400:
            raise SourceError(f"Europe PMC HTTP {resp.status_code}")
        papers = []
        for item in (resp.json().get("resultList") or {}).get("result", []):
            try:
                title = _strip_markup(item.get("title"))
                if not title:
                    continue
                pmcid = item.get("pmcid")
                pdf = None
                for link in ((item.get("fullTextUrlList") or {}).get("fullTextUrl") or []):
                    if link.get("documentStyle") == "pdf" and link.get("availabilityCode") in ("OA", "F"):
                        pdf = link.get("url")
                        break
                source_ids = {"europepmc": f"{item.get('source')}:{item.get('id')}"}
                if item.get("pmid"):
                    source_ids["pmid"] = str(item["pmid"])
                if pmcid:
                    source_ids["pmcid"] = pmcid
                papers.append(Paper(
                    paper_id=f"epmc:{item.get('source')}:{item.get('id')}",
                    title=title,
                    authors=[a.strip() for a in (item.get("authorString") or "").rstrip(".").split(",") if a.strip()][:25],
                    year=int(item["pubYear"]) if str(item.get("pubYear", "")).isdigit() else None,
                    venue=_clean(((item.get("journalInfo") or {}).get("journal") or {}).get("title")) or None,
                    doi=item.get("doi"),
                    url=f"https://europepmc.org/article/{item.get('source')}/{item.get('id')}",
                    pdf_url=pdf,
                    abstract=_strip_markup(item.get("abstractText")) or None,
                    is_open_access=item.get("isOpenAccess") == "Y",
                    oa_status="open" if item.get("isOpenAccess") == "Y" else None,
                    citation_count=item.get("citedByCount"),
                    sources=["europepmc"],
                    source_ids=source_ids,
                ))
            except Exception as exc:  # noqa: BLE001
                log.debug("skipping malformed Europe PMC record: %s", exc)
        return papers


# --------------------------------------------------------------------------- demo fixtures
class FixtureSource(LiteratureSource):
    """Curated real-paper corpus used by DEMO_MODE (see scripts/build_demo_corpus.py)."""

    name = "demo_corpus"

    def __init__(self, path: Path):
        self.path = path

    def search(self, query: str, limit: int) -> list[Paper]:
        data = json.loads(self.path.read_text())
        papers = []
        for item in data["papers"]:
            arxiv = item["source_ids"]["arxiv"]
            papers.append(Paper(paper_id=f"arxiv:{arxiv}", sources=["demo_corpus"], **{
                k: v for k, v in item.items() if k in Paper.model_fields and k not in ("paper_id", "sources")
            }))
        return papers[:limit]
