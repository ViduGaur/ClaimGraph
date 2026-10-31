"""Open-access acquisition: find a legal full-text copy, download it, extract text.

Order of preference for each paper:
1. the OA PDF URL reported by the metadata source (OpenAlex / S2 / Europe PMC)
2. Unpaywall lookup by DOI (only when CLAIMGRAPH_CONTACT_EMAIL is set — required by Unpaywall)
3. arXiv PDF when an arXiv id is known
4. Europe PMC full-text XML (open-access subset; resolved from the DOI) — structured sections, no PDF parsing
5. fall back to the abstract

Only URLs surfaced by OA metadata services are fetched (no crawling), robots.txt
is honoured for non-arXiv hosts, downloads are size-capped and cached on disk.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from app.extraction.text import Section, jats_to_sections, pdf_to_sections
from app.literature.http import PoliteClient, SourceError
from app.schemas.domain import Paper

log = logging.getLogger(__name__)
TRUSTED_HOSTS = ("arxiv.org", "export.arxiv.org", "europepmc.org", "www.ncbi.nlm.nih.gov")


@dataclass
class PaperText:
    paper: Paper
    sections: list[Section] = field(default_factory=list)


def reconcile_abstract(sections: list[Section], abstract: str | None) -> list[Section]:
    """Prefer the clean metadata abstract over the PDF's (often mangled) one.

    PDF heading detection can yield a truncated "abstract" or one that swallows
    the introduction; the latter is kept as introduction text instead.
    """
    if not abstract:
        return sections
    out = [Section("abstract", abstract)]
    for s in sections:
        if s.name != "abstract":
            out.append(s)
        elif len(s.text) > 1.5 * len(abstract):
            out.append(Section("introduction", s.text))
    return out


class Acquirer:
    def __init__(self, client: PoliteClient, cache_dir: Path, contact_email: str = "", max_mb: int = 25,
                 download: bool = True):
        self.client = client
        self.cache_dir = cache_dir
        self.contact_email = contact_email
        self.max_bytes = max_mb * 1_000_000
        self.download = download
        self._robots: dict[str, RobotFileParser | None] = {}

    # ------------------------------------------------------------------ helpers
    def _allowed(self, url: str) -> bool:
        host = urlparse(url).netloc
        if any(host.endswith(t) for t in TRUSTED_HOSTS):
            return True  # arXiv/EPMC explicitly permit programmatic PDF access at polite rates
        if host not in self._robots:
            parser = RobotFileParser()
            try:
                resp = self.client.get(f"{urlparse(url).scheme}://{host}/robots.txt")
                parser.parse(resp.text.splitlines() if resp.status_code == 200 else [])
                self._robots[host] = parser
            except SourceError:
                self._robots[host] = None
        parser = self._robots[host]
        return True if parser is None else parser.can_fetch("ClaimGraph", url)

    def _unpaywall_pdf(self, doi: str) -> str | None:
        if not self.contact_email or not doi:
            return None
        try:
            resp = self.client.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": self.contact_email})
        except SourceError:
            return None
        if resp.status_code != 200:
            return None
        best = (resp.json() or {}).get("best_oa_location") or {}
        return best.get("url_for_pdf")

    EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"

    def _europepmc_sections(self, paper: Paper) -> list[Section] | None:
        """Open-access full text from Europe PMC as JATS XML, if the paper is in its OA subset."""
        pmcid = paper.source_ids.get("pmcid")
        if not pmcid and paper.doi:
            try:
                resp = self.client.get(f"{self.EPMC}/search", params={"query": f'DOI:"{paper.doi}"', "format": "json", "resultType": "lite"})
                hits = (resp.json().get("resultList") or {}).get("result", []) if resp.status_code == 200 else []
            except (SourceError, ValueError):
                return None
            hit = next((h for h in hits if h.get("pmcid") and h.get("isOpenAccess") == "Y"), None)
            pmcid = hit.get("pmcid") if hit else None
        if not pmcid:
            return None
        key = self.cache_dir / f"{pmcid}.xml"
        try:
            if key.exists():
                xml = key.read_bytes()
            else:
                resp = self.client.get(f"{self.EPMC}/{pmcid}/fullTextXML")
                if resp.status_code != 200:
                    return None
                xml = resp.content
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                key.write_bytes(xml)
            return jats_to_sections(xml)
        except (SourceError, ValueError) as exc:
            log.info("Europe PMC full text unavailable for %s: %s", pmcid, exc)
            return None

    def _fetch_pdf(self, url: str) -> bytes:
        key = hashlib.sha1(url.encode()).hexdigest()
        path = self.cache_dir / f"{key}.pdf"
        if path.exists() and path.stat().st_size > 1000:
            return path.read_bytes()
        if not self._allowed(url):
            raise SourceError("disallowed by robots.txt")
        content, ctype = self.client.stream_bytes(url, self.max_bytes)
        if not content.startswith(b"%PDF"):
            raise SourceError(f"not a PDF (got {ctype or 'unknown content'})")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return content

    # ------------------------------------------------------------------ main
    def candidate_urls(self, paper: Paper) -> list[str]:
        urls: list[str] = []
        if paper.pdf_url:
            urls.append(paper.pdf_url)
        arxiv = paper.source_ids.get("arxiv")
        if arxiv:
            urls.append(f"https://arxiv.org/pdf/{arxiv}")
        seen: set[str] = set()
        return [u for u in urls if not (u in seen or seen.add(u))]

    def acquire(self, paper: Paper) -> PaperText:
        notes: list[str] = []
        if self.download:
            urls = self.candidate_urls(paper)
            if not urls and paper.doi:
                found = self._unpaywall_pdf(paper.doi)
                if found:
                    urls.append(found)
            for url in urls:
                try:
                    sections = pdf_to_sections(self._fetch_pdf(url))
                except (SourceError, ValueError) as exc:
                    notes.append(f"{urlparse(url).netloc}: {exc}")
                    continue
                except Exception as exc:  # noqa: BLE001 — never let one paper kill the run
                    log.exception("unexpected error acquiring %s", url)
                    notes.append(f"{urlparse(url).netloc}: unexpected {exc.__class__.__name__}")
                    continue
                sections = reconcile_abstract(sections, paper.abstract)
                return PaperText(paper.model_copy(update={"text_source": "full_text", "acquisition_note": f"full text from {urlparse(url).netloc}"}), sections)
            sections = self._europepmc_sections(paper)
            if sections:
                sections = reconcile_abstract(sections, paper.abstract)
                return PaperText(paper.model_copy(update={"text_source": "full_text", "acquisition_note": "full text from Europe PMC (JATS XML)"}), sections)
            if not urls:
                notes.append("no open-access PDF located")
        if paper.abstract:
            note = "; ".join(notes) or ("PDF download disabled" if not self.download else "abstract only")
            return PaperText(paper.model_copy(update={"text_source": "abstract", "acquisition_note": f"abstract only ({note})"}),
                             [Section("abstract", paper.abstract)])
        return PaperText(paper.model_copy(update={"text_source": "none", "acquisition_note": "no abstract or open-access text available; " + "; ".join(notes)}), [])
