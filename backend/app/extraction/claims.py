"""Claim extraction and normalisation.

Per paper:
1. **Relevant passage retrieval** — every section is chunked into passages that are
   embedded and ranked against the research question; only the abstract plus the
   top passages are mined (this is what keeps the system from "reading" noise).
2. **Claim detection** — sentences are scored with transparent linguistic cues
   (reporting verbs, comparatives, quantitative results, hedges, limitation and
   future-work markers), weighted by section and by relevance to the question.
3. **Structuring** — each claim gets a type, polarity, subject, reported result,
   and a :class:`ClaimContext`; its evidence is the verbatim source passage.
4. **Optional LLM extraction** — when enabled, an LLM proposes claims from the
   same retrieved passages; every LLM claim must fuzzy-match its cited passage
   (grounding check) or it is discarded.
5. **Normalisation / deduplication** — aliases are canonicalised and
   near-duplicate claims within a paper are merged.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz

from app.extraction.acquisition import PaperText
from app.extraction.context import ENTITY_ALIASES, canonical_entities, extract_context, merge_context
from app.extraction.text import SECTION_WEIGHT, Chunk, chunk_section
from app.providers.llm import BudgetedLLM
from app.providers.models import ModelBundle
from app.schemas.domain import Claim, ClaimContext, ClaimType, Paper, Passage, Polarity

log = logging.getLogger(__name__)

_I = re.I
FINDING = re.compile(r"\bwe (show|find|found|observe|demonstrate|report|conclude|confirm|establish|reveal)|"
                     r"\bresults? (show|indicate|suggest|demonstrate|reveal|confirm)|\bour (results|experiments|analysis|findings|study) "
                     r"(show|indicate|suggest|demonstrate|reveal)|\b(this|which) (leads|results) (us )?to|\bexperimental results show|"
                     r"\bwe (also )?(empirically )?(analy[sz]e|verify)|\bsurprisingly\b|\bnotably\b|\bas one result\b", _I)
COMPARATIVE = re.compile(r"outperform\w*|underperform\w*|surpass\w*|better than|worse than|superior|inferior|"
                         r"compete (favorably|favourably)|competitive (with|results)|comparable (to|with|accuracy|performance)|on par|"
                         r"\bmatch(es|ed)?\b|similar performance|improv(e|es|ed|ement)s?\b|degrad\w*|exceed\w*|\bbeat\w*|"
                         r"more data-?hungry|less data|attains? excellent|perform (very )?well|excel\b|avoid overfitting|"
                         r"not (competitive|suitable|necessary)|is sufficient|superseded", _I)
RESULT = re.compile(r"[+\-]?\d+(\.\d+)?\s?%|\btop-?1\b|accuracy of|state-of-the-art|\bSOTA\b|significant(ly)?|"
                    r"\d+(\.\d+)?x (smaller|faster|fewer)|scaling law", _I)
BACKGROUND = re.compile(r"^(recently|in recent years|over the past)|it is (commonly|widely|generally) (accepted|believed|assumed)|"
                        r"many (researchers |have come to )?believe|is generally found|(have|has) been (shown|found|reported|proposed)|"
                        r"existing works?|previous (work|studies|methods)|prior work|is interpreted as|were shown to|"
                        r"it is well[- ]known|while the transformer|began with", _I)
METHOD = re.compile(r"\bwe (propose|introduce|present|develop|design|conduct|consider|train|gradually)\b|this (paper|work|note) "
                    r"(proposes|introduces|presents|aims|investigates)|in this (paper|work), we (propose|introduce|present|aim|reexamine)|"
                    r"\bwe aim\b|\bour (method|approach|model|task) (is|can|does)", _I)
LIMITATION = re.compile(r"\blimitations?\b|\bfail(s|ed)? to\b|\bcannot\b|\bstruggle|\bdoes not generali[sz]e|"
                        r"\bremains? (challenging|difficult|limited|unclear)|\bdifficult(y|ies)\b|\bsensitiv(e|ity) to|"
                        r"\bless robust|\bnot suitable|\bdata-?hungry|\breli(ance|es) on|\bdependence on|\blimiting\b|"
                        r"optimization-related problems", _I)
FUTURE = re.compile(r"future (work|research|studies|directions)|further (research|work|investigation|studies)|"
                    r"remains an open|we leave .{0,60}(to|for) future|open (question|problem)", _I)
HEDGE = re.compile(r"\b(may|might|could|possibly|potentially|appears? to|seems? to|suggests?|likely|preliminary|sometimes)\b", _I)
BOILERPLATE = re.compile(r"code (is|and (pre-trained )?models are) (publicly )?available|github\.com|https?://|\\url|"
                         r"we (share|release) (our )?code|checkpoints are available|is organized as follows|in section \d|"
                         r"^table \d|^figure \d", _I)

POSITIVE = re.compile(r"outperform\w*|surpass\w*|better than|superior|improv(e|es|ed)\b|attains? excellent|perform (very )?well|"
                      r"excel\w*|\bbeat\w*|avoid overfitting|compete (favorably|favourably)|state-of-the-art|new SOTA|"
                      r"is sufficient|more robust|effective|superseded|dramatically improve|significant improvement|"
                      r"not necessary|much more robust", _I)
NEGATIVE = re.compile(r"underperform\w*|worse than|inferior|fail(s|ed)?\b|not (competitive|suitable)|data-?hungry|"
                      r"difficult(y|ies)|lack of|requires? (large|massive|sophisticated)|reli(ance|es) on (large|massive)|"
                      r"dependence on a large|sensitiv(e|ity) to|limit(ed|ing|s) (its|their)? ?adoption|degrad\w*|"
                      r"optimization-related problems|weaker inductive bias|increased reliance", _I)
PARITY = re.compile(r"\bmatch(es|ed)?\b|comparable|on par|similar (performance|accuracy)|competitive (with|results)|"
                    r"same performance|either match", _I)

FAMILY = {"ViT": "transformer", "DeiT": "transformer", "Swin Transformer": "transformer", "CCT": "transformer",
          "Transformer": "transformer", "MLP-Mixer": "mlp", "ConvNeXt": "cnn", "NFNet": "cnn", "ResNet": "cnn",
          "CNN": "cnn"}
_FRAMES = re.compile(r"^(we (show|find|found|observe|demonstrate|report|conclude|confirm) (that )?|"
                     r"(our |experimental )?results (show|indicate|suggest|demonstrate) (that )?|surprisingly, |notably, |"
                     r"as one result of this study,? we find (that )?|moreover, |however, |in fact, |hence, )", _I)
_ALIAS_RES = [(name, re.compile(pat, re.I)) for name, pat in ENTITY_ALIASES[:10]]
_ENTITY_POS_RES = [(name, re.compile(pat, re.I)) for name, pat in ENTITY_ALIASES]


def family(entity: str | None) -> str | None:
    if not entity:
        return None
    return FAMILY.get(entity, entity.lower())


def normalize_claim_text(text: str) -> str:
    """Strip reporting frames/hedges and canonicalise entity aliases (used for comparison)."""
    out = text.strip()
    for _ in range(3):
        new = _FRAMES.sub("", out)
        if new == out:
            break
        out = new
    for name, rx in _ALIAS_RES:
        out = rx.sub(name, out)
    out = re.sub(r"\b([\w\-]+(?: [\w\-]+){0,2}) \(\1\)", r"\1", out)  # "ViT (ViT)" after alias replacement
    out = re.sub(r"\s+", " ", out).strip(" .")
    return out[:1].upper() + out[1:] if out else text


def detect_polarity(text: str) -> Polarity:
    pos = len(POSITIVE.findall(text))
    neg = len(NEGATIVE.findall(text))
    negated_pos = len(re.findall(r"\b(not|no|never|without)\s+(\w+\s+){0,2}(outperform|improv|surpass|excel|better)", text, _I))
    pos, neg = pos - negated_pos, neg + negated_pos
    if pos > neg:
        return Polarity.POSITIVE
    if neg > pos:
        return Polarity.NEGATIVE
    if PARITY.search(text):
        return Polarity.NEUTRAL
    return Polarity.NONE


def classify_type(sentence: str, section: str) -> ClaimType | None:
    if BOILERPLATE.search(sentence):
        return None
    if FUTURE.search(sentence) or section == "conclusion" and re.search(r"\bfuture\b", sentence, _I):
        return ClaimType.FUTURE_WORK
    if BACKGROUND.search(sentence) or section == "related_work":
        return ClaimType.BACKGROUND
    comparative = COMPARATIVE.search(sentence)
    result = RESULT.search(sentence)
    finding = FINDING.search(sentence)
    if METHOD.search(sentence) and not (comparative or result):
        return ClaimType.METHOD
    if LIMITATION.search(sentence) and (section == "limitations" or re.search(r"\b(however|but|although|yet|still)\b", sentence, _I)) and not finding:
        return ClaimType.LIMITATION
    if comparative:
        return ClaimType.COMPARISON
    if finding or result:
        return ClaimType.FINDING
    if LIMITATION.search(sentence):
        return ClaimType.LIMITATION
    return None


_HEADING_RESIDUE = re.compile(r"^(?:[A-Z][\w\-]*\s){1,5}(?=(?:The|We|Our|In|This|These|For|On|As|To)\s[a-z])")


def strip_heading_residue(sentence: str) -> str:
    """'Arithmetic Reasoning The results show …' → 'The results show …' (PDF headings glued to text)."""
    return _HEADING_RESIDUE.sub("", sentence, count=1)


def is_clean_sentence(sentence: str) -> bool:
    """Reject PDF-extraction debris: table rows, merged columns, heading run-ons."""
    tokens = sentence.split()
    alpha = sum(1 for t in tokens if re.fullmatch(r"[A-Za-z][A-Za-z\-’'()]*[.,;:]?", t))
    if alpha / max(1, len(tokens)) < 0.7:
        return False
    if sum(1 for t in tokens if len(t) == 1 and t.isalpha() and t not in ("a", "A", "I")) > 2:
        return False
    if re.search(r"\b(Table|Figure|Fig\.)\s*\d+[.:]?\s*$|^\W|^\(?[a-z]\)", sentence):
        return False
    return not re.search(r"[a-z]{3,}[A-Z][a-z]{3,}", sentence.replace("CoT", ""))  # glued words from merged columns


def cue_score(sentence: str, ctype: ClaimType) -> float:
    s = 0.0
    s += 0.35 if FINDING.search(sentence) else 0.0
    s += 0.30 if COMPARATIVE.search(sentence) else 0.0
    s += 0.20 if RESULT.search(sentence) else 0.0
    s += {ClaimType.LIMITATION: 0.25, ClaimType.FUTURE_WORK: 0.25, ClaimType.BACKGROUND: 0.10, ClaimType.METHOD: 0.05}.get(ctype, 0.0)
    return min(s, 0.85)


def _positions(text: str) -> list[tuple[int, str]]:
    """Entity mentions with their offsets; earlier (more specific) aliases win overlapping spans."""
    hits: list[tuple[int, str]] = []
    spans: list[tuple[int, int]] = []
    for name, rx in _ENTITY_POS_RES:
        for m in rx.finditer(text):
            if any(s < m.end() and m.start() < e for s, e in spans):
                continue
            spans.append((m.start(), m.end()))
            hits.append((m.start(), name))
    return sorted(hits)


def subject_and_object(sentence: str, title: str = "") -> tuple[str | None, str | None]:
    """Subject = entity nearest *before* the comparative/result verb, object = first entity after it.

    Sentences about "our model" with no named entity before the verb fall back to
    the main entity in the paper title (e.g. "Compact Transformers" → CCT).
    """
    hits = _positions(sentence)
    verb = COMPARATIVE.search(sentence) or RESULT.search(sentence) or LIMITATION.search(sentence)
    title_ents = canonical_entities(title) if title else []
    if verb:
        before = [n for p, n in hits if p < verb.start()]
        after = [n for p, n in hits if p >= verb.start()]
        subject = before[-1] if before else None
        if subject is None and re.search(r"\b(our|we|these|this|its|their|the resultant)\b", sentence[: verb.start()], _I):
            subject = title_ents[0] if title_ents else None
        obj = next((n for n in after if n != subject and family(n) != family(subject)), None)
        if subject is None and hits:
            subject = hits[0][1]
        return subject, obj
    return (hits[0][1] if hits else (title_ents[0] if title_ents else None)), None


def reported_result(sentence: str) -> str | None:
    spans = re.findall(r"(?:[+\-]?\d+(?:\.\d+)?\s?%(?:\s+(?:top-?1\s+)?accuracy)?|top-?1 accuracy of \d+(?:\.\d+)?%|"
                       r"\d+(?:\.\d+)?x (?:smaller|faster|fewer \w+))", sentence, _I)
    return ", ".join(dict.fromkeys(s.strip() for s in spans)) or None


def _hash(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode()).hexdigest()[:12]


@dataclass
class _Candidate:
    sentence: str
    chunk: Chunk
    index_in_chunk: int
    ctype: ClaimType
    score: float
    relevance: float


LLM_CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "passage": {"type": "integer"},
                    "quote": {"type": "string"},
                    "claim_type": {"type": "string", "enum": [t.value for t in ClaimType]},
                    "hedged": {"type": "boolean"},
                },
                "required": ["passage", "quote", "claim_type"],
            },
        }
    },
    "required": ["claims"],
}
LLM_SYSTEM = (
    "You extract scientific claims from paper passages. A claim is a statement the paper itself asserts "
    "(a finding, comparison, limitation, or future-work statement) that is relevant to the research question. "
    "Copy each claim VERBATIM as a single sentence from the numbered passage it comes from. Never invent text. "
    "Return at most 5 claims."
)


SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "keep": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"},
                               "type": {"type": "string", "enum": ["finding", "comparison", "limitation"]}},
                "required": ["n", "type"],
            },
        }
    },
    "required": ["keep"],
}
SELECT_SYSTEM = (
    "You screen numbered sentences taken verbatim from ONE scientific paper. Select the sentences that state a "
    "result, finding or conclusion of THIS paper that bears on the research question. Keep a sentence only if all "
    "of these hold: (1) it asserts something this paper itself found or concludes - not background attributed to "
    "other work, not a description of what was done or will be done, not a question, not a heading fragment or "
    "garbled text; (2) it is relevant to the research question; (3) a reader can tell what is being asserted about "
    "what. Prefer specific findings over vague ones and skip near-duplicates. Return the numbers of at most "
    "{k} sentences, best first, each with its type: finding (a result), comparison (X versus Y) or limitation. "
    "Return an empty list if none qualify."
)


class ClaimExtractor:
    def __init__(self, models: ModelBundle, llm: BudgetedLLM, max_claims_per_paper: int = 6,
                 use_llm_extraction: bool = False, top_passages: int = 10, llm_selection: bool = False,
                 selection_pool: int = 24):
        self.models = models
        self.llm = llm
        self.max_claims = max_claims_per_paper
        self.use_llm = use_llm_extraction
        self.top_passages = top_passages
        # LLM as a *selector* over verbatim candidate sentences: it can drop or keep, never rewrite
        self.llm_selection = llm_selection
        self.selection_pool = selection_pool

    # ------------------------------------------------------------------ retrieval
    def retrieve_passages(self, question_vec: np.ndarray, doc: PaperText) -> list[tuple[Chunk, float]]:
        chunks: list[Chunk] = []
        for section in doc.sections:
            chunks.extend(chunk_section(section))
        if not chunks:
            return []
        vecs = self.models.embedder.embed([c.text for c in chunks])
        sims = vecs @ question_vec
        scored = [(c, float(s) * SECTION_WEIGHT.get(c.section, 0.6)) for c, s in zip(chunks, sims)]
        abstract = [(c, s) for c, s in scored if c.section == "abstract"]
        others = sorted([(c, s) for c, s in scored if c.section != "abstract"], key=lambda x: x[1], reverse=True)
        return abstract + others[: self.top_passages]

    # ------------------------------------------------------------------ extraction
    def extract(self, question: str, question_vec: np.ndarray, doc: PaperText) -> tuple[list[Claim], list[str]]:
        warnings: list[str] = []
        passages = self.retrieve_passages(question_vec, doc)
        if not passages:
            return [], [f"{doc.paper.citation}: no text to extract claims from"]
        paper_ctx = extract_context(" ".join(s.text for s in doc.sections if s.name in ("abstract", "method", "experiments"))[:20000])
        claims: list[Claim] = []
        if self.use_llm and self.llm.enabled:
            claims = self._extract_llm(question, doc.paper, passages, paper_ctx)
            if not claims:
                warnings.append(f"{doc.paper.citation}: LLM extraction unavailable/ungrounded, used rule-based extraction")
        if not claims:
            claims = self._extract_rules(question_vec, doc.paper, passages, paper_ctx, question)
        return self._dedupe(claims), warnings

    def _select_with_llm(self, question: str, paper: Paper, cands: list[_Candidate]) -> list[_Candidate] | None:
        """Let the LLM pick which candidate sentences are real claims. ``None`` → keep the rule ranking."""
        pool = cands[: self.selection_pool]
        listing = "\n".join(f"[{i}] ({c.chunk.section}) {c.sentence}" for i, c in enumerate(pool))
        user = f"Research question: {question}\nPaper: {paper.title} ({paper.year})\n\nSentences:\n{listing}"
        out = self.llm.complete_json(SELECT_SYSTEM.format(k=self.max_claims), user, SELECT_SCHEMA, max_tokens=300)
        if out is None or not isinstance(out.get("keep"), list):
            return None
        chosen: list[_Candidate] = []
        for item in out["keep"]:
            try:
                idx, ctype = int(item["n"]), ClaimType(item["type"])
            except (KeyError, ValueError, TypeError):
                continue
            if 0 <= idx < len(pool) and pool[idx] not in chosen:
                pool[idx].ctype = ctype
                chosen.append(pool[idx])
        return chosen[: self.max_claims]

    def _extract_rules(self, question_vec: np.ndarray, paper: Paper, passages: list[tuple[Chunk, float]],
                       paper_ctx: ClaimContext, question: str = "") -> list[Claim]:
        cands: list[_Candidate] = []
        for chunk, _ in passages:
            for i, raw in enumerate(chunk.sentences):
                sent = strip_heading_residue(raw)
                words = len(sent.split())
                if words < 7 or words > 80 or not is_clean_sentence(sent):
                    continue
                ctype = classify_type(sent, chunk.section)
                if ctype is None:
                    continue
                cands.append(_Candidate(sent, chunk, i, ctype, 0.0, 0.0))
        if not cands:
            return []
        # overlapping chunks repeat sentences — keep first occurrence
        uniq: dict[str, _Candidate] = {}
        for c in cands:
            uniq.setdefault(c.sentence, c)
        cands = list(uniq.values())
        vecs = self.models.embedder.embed([c.sentence for c in cands])
        sims = vecs @ question_vec
        lo, hi = float(sims.min()), float(sims.max())
        for c, s in zip(cands, sims):
            rel = (float(s) - lo) / (hi - lo) if hi > lo else 0.5
            c.relevance = float(s)
            c.score = (cue_score(c.sentence, c.ctype) + 0.45 * rel) * SECTION_WEIGHT.get(c.chunk.section, 0.6)
        cands.sort(key=lambda c: c.score, reverse=True)
        if self.llm_selection and self.llm.enabled:
            selected = self._select_with_llm(question, paper, cands)
            if selected is not None:
                return [self._build(paper, c.sentence, c.chunk, c.index_in_chunk, c.ctype, c.relevance, c.score, paper_ctx,
                                    "llm_selected", 1.0) for c in selected]
        chosen = [c for c in cands if c.score >= 0.3 and c.ctype != ClaimType.METHOD][: self.max_claims]
        # keep one METHOD claim per paper for context if room remains
        methods = [c for c in cands if c.ctype == ClaimType.METHOD]
        if methods and len(chosen) < self.max_claims:
            chosen.append(methods[0])
        return [self._build(paper, c.sentence, c.chunk, c.index_in_chunk, c.ctype, c.relevance, c.score, paper_ctx, "heuristic", 1.0)
                for c in chosen]

    def _build(self, paper: Paper, sentence: str, chunk: Chunk, idx: int, ctype: ClaimType, relevance: float,
               score: float, paper_ctx: ClaimContext, method: str, grounding: float, hedged: bool | None = None) -> Claim:
        lo, hi = max(0, idx - 1), min(len(chunk.sentences), idx + 2)
        window = chunk.sentences[lo:hi]
        passage_text = " ".join(window)
        passage = Passage(
            passage_id=f"psg_{_hash(paper.paper_id, passage_text)}",
            paper_id=paper.paper_id,
            section=chunk.section,
            text=passage_text,
            relevance=round(relevance, 4),
        )
        ctx_sentence = extract_context(sentence)
        ctx_window = extract_context(passage_text)
        ctx, inherited_local = merge_context(ctx_sentence, ctx_window)
        ctx, inherited_paper = merge_context(ctx, paper_ctx)
        notes = []
        if inherited_paper:
            notes.append("context fields inherited from paper-level text: " + ", ".join(inherited_paper))
        is_hedged = hedged if hedged is not None else bool(HEDGE.search(sentence))
        conf = 0.45 + 0.4 * min(1.0, score) + (0.1 if RESULT.search(sentence) else 0.0)
        if is_hedged:
            conf -= 0.15
            notes.append("claim is hedged in the source (may/might/suggest)")
        if paper.text_source == "abstract":
            conf -= 0.05
            notes.append("extracted from abstract only; full text not available")
        if ctype == ClaimType.BACKGROUND:
            conf = min(conf, 0.5)
            notes.append("statement attributed to prior work / the field, not a new finding of this paper")
        conf *= grounding
        subject, obj = subject_and_object(sentence, paper.title)
        if obj and obj not in ctx.baselines:
            ctx = ctx.model_copy(update={"baselines": [obj, *ctx.baselines]})
        return Claim(
            claim_id=f"clm_{_hash(paper.paper_id, sentence)}",
            paper_id=paper.paper_id,
            claim_text=sentence,
            normalized_text=normalize_claim_text(sentence),
            claim_type=ctype,
            polarity=detect_polarity(sentence),
            subject=subject,
            evidence=[passage],
            section=chunk.section,
            context=ctx,
            inherited_context=inherited_paper,
            reported_result=reported_result(sentence),
            confidence=round(max(0.05, min(0.95, conf)), 3),
            extraction_method=method,  # type: ignore[arg-type]
            grounding_score=round(grounding, 3),
            source_citation=paper.citation,
            relevance=round(relevance, 4),
            uncertainty_notes=notes,
        )

    def _extract_llm(self, question: str, paper: Paper, passages: list[tuple[Chunk, float]], paper_ctx: ClaimContext) -> list[Claim]:
        top = passages[:6]
        listing = "\n".join(f"[{i}] ({c.section}) {c.text[:900]}" for i, (c, _) in enumerate(top))
        user = f"Research question: {question}\nPaper: {paper.title} ({paper.year})\n\nPassages:\n{listing}"
        out = self.llm.complete_json(LLM_SYSTEM, user, LLM_CLAIM_SCHEMA, max_tokens=500)
        if not out:
            return []
        claims = []
        for item in (out.get("claims") or [])[: self.max_claims]:
            try:
                idx = int(item.get("passage", -1))
                quote = str(item.get("quote", "")).strip()
                ctype = ClaimType(item.get("claim_type", "finding"))
            except (ValueError, TypeError):
                continue
            if not (0 <= idx < len(top)) or len(quote) < 20:
                continue
            chunk, rel = top[idx]
            # grounding: locate the best-matching source sentence for the quoted claim
            best_i, best = max(((i, fuzz.ratio(quote.lower(), s.lower())) for i, s in enumerate(chunk.sentences)), key=lambda x: x[1])
            partial = fuzz.partial_ratio(quote.lower(), chunk.text.lower())
            grounding = max(best, partial) / 100
            if grounding < 0.6:
                log.info("dropping ungrounded LLM claim (%.2f): %s", grounding, quote[:80])
                continue
            sentence = chunk.sentences[best_i] if best >= 70 else quote
            claims.append(self._build(paper, sentence, chunk, best_i, ctype, rel, 0.7, paper_ctx, "llm", grounding,
                                      hedged=bool(item.get("hedged"))))
        return claims

    def _dedupe(self, claims: list[Claim]) -> list[Claim]:
        if len(claims) < 2:
            return claims
        vecs = self.models.embedder.embed([c.normalized_text for c in claims])
        keep: list[int] = []
        for i in sorted(range(len(claims)), key=lambda k: claims[k].confidence, reverse=True):
            if all(float(vecs[i] @ vecs[j]) < 0.93 and claims[i].claim_text != claims[j].claim_text for j in keep):
                keep.append(i)
        return [claims[i] for i in sorted(keep)]

