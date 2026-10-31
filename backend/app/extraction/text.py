"""Text utilities: PDF → sections → sentences → passages."""
from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- sentences
_ABBREV = ["e.g.", "i.e.", "et al.", "etc.", "Fig.", "Figs.", "Eq.", "Eqs.", "vs.", "cf.", "approx.", "resp.",
           "Sec.", "Tab.", "No.", "Dr.", "al.", "ca.", "incl.", "w.r.t."]
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Rule-based splitter that survives abbreviations, decimals and initials."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    protected = text
    for i, ab in enumerate(_ABBREV):
        protected = protected.replace(ab, ab.replace(".", f"§{i}§"))
    protected = re.sub(r"(\d)\.(\d)", r"\1§d§\2", protected)
    protected = re.sub(r"\b([A-Z])\.(?=\s?[A-Z])", r"\1§i§", protected)
    parts = _SPLIT.split(protected)
    out = []
    for part in parts:
        restored = part.replace("§d§", ".").replace("§i§", ".")
        restored = re.sub(r"§\d+§", ".", restored)
        if restored.strip():
            out.append(restored.strip())
    return out


# --------------------------------------------------------------------------- sections
SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("abstract", re.compile(r"^abstract\b", re.I)),
    ("introduction", re.compile(r"^(introduction|motivation)\b", re.I)),
    ("related_work", re.compile(r"^(related work|background|prior work|literature review|preliminaries)\b", re.I)),
    ("method", re.compile(r"^(method|methods|methodology|approach|proposed method|materials and methods|our approach)\b", re.I)),
    ("experiments", re.compile(r"^(experiment|experiments|experimental (setup|settings?|design)|implementation details|setup|evaluation)\b", re.I)),
    ("results", re.compile(r"^(results?|main results|findings|empirical results)\b", re.I)),
    ("discussion", re.compile(r"^(discussion|analysis|ablation)", re.I)),
    ("limitations", re.compile(r"^(limitations?|threats to validity)\b", re.I)),
    ("conclusion", re.compile(r"^(conclusions?|concluding remarks|summary|future work)\b", re.I)),
    ("references", re.compile(r"^(references|bibliography|acknowledge?ments?|appendix|supplementary)\b", re.I)),
]
_HEADING = re.compile(r"^\s*(?:[IVX]+\.|\d+(?:\.\d+)*\.?)?\s*([A-Z][A-Za-z][A-Za-z &\-:]{1,60})\s*$")

# how much a claim found in a section is worth, relative to the abstract
SECTION_WEIGHT = {
    "abstract": 1.0, "results": 1.0, "conclusion": 0.95, "discussion": 0.9, "experiments": 0.85,
    "limitations": 0.85, "introduction": 0.6, "method": 0.5, "related_work": 0.25, "body": 0.6,
}


def classify_heading(line: str) -> str | None:
    m = _HEADING.match(line)
    if not m or len(line.split()) > 8:
        return None
    title = m.group(1).strip()
    for name, pat in SECTION_PATTERNS:
        if pat.match(title):
            return name
    return None


@dataclass
class Section:
    name: str
    text: str


def sections_from_text(raw: str) -> list[Section]:
    """Split plain text (e.g. from a PDF) into canonical sections; drops references onwards."""
    raw = re.sub(r"-\n(?=[a-z])", "", raw)  # de-hyphenate line breaks
    lines = raw.splitlines()
    sections: list[Section] = []
    current, buf = "body", []

    def flush() -> None:
        text = re.sub(r"\s+", " ", " ".join(buf)).strip()
        if text:
            sections.append(Section(current, text))

    for line in lines:
        stripped = line.strip()
        name = classify_heading(stripped) if 2 < len(stripped) < 70 else None
        if name:
            flush()
            buf = []
            if name == "references":
                return _merge_same(sections)
            current = name
            continue
        buf.append(stripped)
    flush()
    return _merge_same(sections)


def _merge_same(sections: list[Section]) -> list[Section]:
    merged: list[Section] = []
    for s in sections:
        if merged and merged[-1].name == s.name:
            merged[-1] = Section(s.name, merged[-1].text + " " + s.text)
        else:
            merged.append(s)
    return merged


def _looks_like_figure_or_table(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    chars = "".join(lines)
    digits = sum(ch.isdigit() for ch in chars) / max(1, len(chars))
    avg = sum(len(ln) for ln in lines) / len(lines)
    return digits > 0.3 or (len(lines) >= 3 and avg < 28)


def _strip_running_headers(pages: list[str]) -> list[str]:
    """Drop short lines that repeat on many pages (journal running heads, footers, page numbers)."""
    if len(pages) < 3:
        return pages
    norm = lambda line: re.sub(r"\d+", "#", line.strip().lower())  # noqa: E731 — page numbers vary
    counts: dict[str, int] = {}
    for page in pages:
        for key in {norm(ln) for ln in page.splitlines() if 0 < len(ln.strip()) < 120}:
            counts[key] = counts.get(key, 0) + 1
    threshold = max(2, round(0.4 * len(pages)))
    repeated = {k for k, c in counts.items() if c >= threshold}
    return ["\n".join(ln for ln in page.splitlines() if norm(ln) not in repeated) for page in pages]


def _page_text(page) -> str:
    """Reading-order text for one page, handling two-column layouts.

    PyMuPDF's global sort interleaves the two columns line by line. Instead we
    take text blocks, split the page into vertical bands at full-width blocks,
    and within each band read the left column before the right one. Blocks that
    look like figure/table content are dropped.
    """
    w = page.rect.width
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    blocks = [b for b in blocks if not _looks_like_figure_or_table(b[4]) or len(b[4].split()) <= 6]
    narrow_left = sum(1 for b in blocks if b[2] < 0.56 * w and (b[2] - b[0]) < 0.55 * w)
    narrow_right = sum(1 for b in blocks if b[0] > 0.44 * w and (b[2] - b[0]) < 0.55 * w)
    two_col = narrow_left >= 2 and narrow_right >= 2

    def text_of(b) -> str:
        return b[4].strip()

    if not two_col:
        return "\n".join(text_of(b) for b in sorted(blocks, key=lambda b: (round(b[1]), b[0])))
    spanning = sorted((b for b in blocks if (b[2] - b[0]) >= 0.55 * w), key=lambda b: b[1])
    columns = [b for b in blocks if (b[2] - b[0]) < 0.55 * w]
    out: list[str] = []
    edges = [-1.0] + [b[1] for b in spanning] + [float("inf")]
    for i in range(len(edges) - 1):
        top, bottom = edges[i], edges[i + 1]
        band = [b for b in columns if top <= b[1] < bottom]
        left = sorted((b for b in band if b[0] < 0.44 * w), key=lambda b: b[1])
        right = sorted((b for b in band if b[0] >= 0.44 * w), key=lambda b: b[1])
        out.extend(text_of(b) for b in left + right)
        if i < len(spanning):
            out.append(text_of(spanning[i]))
    return "\n".join(out)


def pdf_to_sections(pdf_bytes: bytes) -> list[Section]:
    """Extract text with PyMuPDF and segment it. Raises ValueError for unusable PDFs."""
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("not a PDF file")
    import pymupdf

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"unreadable PDF: {exc}") from exc
    with doc:
        if doc.needs_pass:
            raise ValueError("PDF is encrypted")
        pages = [_page_text(page) for page in doc]
    raw = "\n".join(_strip_running_headers(pages))
    if len(raw.strip()) < 500:
        raise ValueError("PDF has no extractable text (scanned image?)")
    sections = sections_from_text(raw)
    if not sections:
        raise ValueError("no text sections recovered")
    return sections


def jats_to_sections(xml: bytes) -> list[Section]:
    """Europe PMC / PMC full-text JATS XML → canonical sections (references, figures, tables dropped)."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError(f"malformed JATS XML: {exc}") from exc
    drop = ("xref", "table-wrap", "fig", "disp-formula", "inline-formula", "ref-list", "supplementary-material")
    for parent in list(root.iter()):
        children = list(parent)
        for i, child in enumerate(children):
            if child.tag in drop:
                # ElementTree stores the text *after* an element on the element itself (.tail): keep it
                if child.tail:
                    prev = children[i - 1] if i > 0 else None
                    if prev is not None and prev in list(parent):
                        prev.tail = (prev.tail or "") + child.tail
                    else:
                        parent.text = (parent.text or "") + child.tail
                parent.remove(child)

    def text(el) -> str:
        t = re.sub(r"\s+", " ", "".join(el.itertext()))
        t = re.sub(r"[\[(]\s*(?:[,;–\-]\s*)*[\])]", "", t)  # citation brackets emptied by dropping <xref>
        return re.sub(r"\s+([,.;:])", r"\1", re.sub(r"\s+", " ", t)).strip()

    sections: list[Section] = []
    abstract = root.find(".//front//abstract")
    if abstract is not None:
        sections.append(Section("abstract", " ".join(text(p) for p in abstract.iter("p")) or text(abstract)))
    body = root.find(".//body")
    if body is not None:
        for sec in body.findall("./sec"):
            title_el = sec.find("title")
            title = text(title_el) if title_el is not None else ""
            name = classify_heading(re.sub(r"^[\d.\s]+", "", title)) or "body"
            if name == "references":
                continue
            paras = " ".join(text(p) for p in sec.iter("p"))
            if paras:
                sections.append(Section(name, paras))
    sections = _merge_same([s for s in sections if s.text])
    if sum(len(s.text) for s in sections) < 500:
        raise ValueError("JATS document has no usable body text")
    return sections


# --------------------------------------------------------------------------- passages
@dataclass
class Chunk:
    section: str
    sentences: list[str]
    start_sentence: int

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


def chunk_section(section: Section, size: int = 4, overlap: int = 1) -> list[Chunk]:
    sents = split_sentences(section.text)
    chunks, i = [], 0
    while i < len(sents):
        chunks.append(Chunk(section.name, sents[i : i + size], i))
        if i + size >= len(sents):
            break
        i += size - overlap
    return chunks
