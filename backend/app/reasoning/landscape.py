"""Research landscape: themes (consensus / disputed / emerging) and potential gaps.

Themes are communities in a claim graph whose edges are semantic similarity
plus the classified relationships. Gap detection is intentionally conservative:
every gap is labelled "Potential research gap", scoped to "within the analysed
papers", and carries the claim ids that motivated it.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import networkx as nx

from app.extraction.claims import family
from app.providers.llm import BudgetedLLM
from app.providers.models import ModelBundle
from app.schemas.domain import (
    Claim,
    ClaimType,
    ConflictAnalysis,
    ConflictVerdict,
    Landscape,
    Paper,
    RelationType,
    Relationship,
    ResearchGap,
    Theme,
)

_STOP = set("""a an the of on in for to and or with by from as at is are was were be been this that these those we our it its
their they which when while than can may also such more most using used use show shows shown find found results result
model models method methods approach paper work data performance based new first two one both however moreover than
very well even only other into over under via per each same much many further notably surprisingly hence leads report
obtain additionally recently best reach come believe have has terms flexible interest especially show paper note present
presents propose proposed achieve achieves achieving can could""".split())


def _tokens(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z][a-z0-9\-]+", text.lower()) if w not in _STOP and len(w) > 2]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def _keywords(cluster_texts: list[str], all_texts: list[str], k: int = 4) -> list[str]:
    """c-TF-IDF: terms frequent in the cluster but not everywhere."""
    df = Counter(t for text in all_texts for t in set(_tokens(text)))
    tf = Counter(t for text in cluster_texts for t in _tokens(text))
    n = len(all_texts)
    scored = {t: c * math.log(1 + n / df[t]) * (1.3 if " " in t else 1.0) for t, c in tf.items() if c >= 2 or len(cluster_texts) == 1}
    picked: list[str] = []
    for term, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True):
        if any(term in p or p in term for p in picked):
            continue
        picked.append(term)
        if len(picked) == k:
            break
    return picked


def clip(text: str, n: int) -> str:
    """Shorten at a word boundary."""
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _gid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]


class LandscapeBuilder:
    def __init__(self, models: ModelBundle, llm: BudgetedLLM, theme_threshold: float | None = None):
        self.models = models
        self.llm = llm
        self.theme_threshold = theme_threshold or (0.8 if models.embedder.name == "fastembed" else 0.2)

    # ------------------------------------------------------------------ themes
    def themes(self, claims: list[Claim], relationships: list[Relationship], conflicts: list[ConflictAnalysis]) -> list[Theme]:
        pool = [c for c in claims if c.claim_type not in (ClaimType.METHOD, ClaimType.FUTURE_WORK)]
        if not pool:
            return []
        vecs = self.models.embedder.embed([c.normalized_text for c in pool])
        g = nx.Graph()
        g.add_nodes_from(c.claim_id for c in pool)
        sims = vecs @ vecs.T
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                if sims[i, j] >= self.theme_threshold:
                    g.add_edge(pool[i].claim_id, pool[j].claim_id, weight=float(sims[i, j]))
        for r in relationships:
            if r.source_claim_id in g and r.target_claim_id in g:
                w = g.get_edge_data(r.source_claim_id, r.target_claim_id, {}).get("weight", 0)
                g.add_edge(r.source_claim_id, r.target_claim_id, weight=max(w, 0.9))
        communities = nx.algorithms.community.greedy_modularity_communities(g, weight="weight") if g.number_of_edges() else [{n} for n in g]
        by_id = {c.claim_id: c for c in pool}
        all_texts = [c.normalized_text for c in pool]
        conflict_rel_ids = {c.relation_id: c for c in conflicts}
        themes: list[Theme] = []
        for comm in communities:
            members = [by_id[m] for m in comm]
            paper_ids = sorted({c.paper_id for c in members})
            ids = {c.claim_id for c in members}
            internal = [r for r in relationships if r.source_claim_id in ids and r.target_claim_id in ids]
            support = sum(1 for r in internal if r.relation in (RelationType.SUPPORTS, RelationType.EXTENDS))
            conflict = [conflict_rel_ids[r.relation_id] for r in internal if r.relation_id in conflict_rel_ids]
            if len(paper_ids) == 1:
                status = "single_source"
            elif conflict:
                status = "disputed"
            elif support:
                status = "consensus"
            else:
                status = "emerging"
            keywords = _keywords([c.normalized_text for c in members], all_texts)
            label = self._label(members, keywords)
            themes.append(Theme(
                theme_id="th_" + _gid(*sorted(ids)),
                label=label,
                keywords=keywords,
                claim_ids=[c.claim_id for c in sorted(members, key=lambda c: -c.relevance)],
                paper_ids=paper_ids,
                support_edges=support,
                conflict_edges=len(conflict),
                status=status,  # type: ignore[arg-type]
                summary=self._theme_summary(members, paper_ids, support, conflict),
            ))
        themes.sort(key=lambda t: (-len(t.paper_ids), -len(t.claim_ids)))
        return themes

    @staticmethod
    def _label(members: list[Claim], keywords: list[str]) -> str:
        """Subject + the context values most shared by the theme's claims (falls back to keywords)."""
        ents = Counter(c.subject for c in members if c.subject)
        ctx = Counter(v for c in members for v in (*c.context.conditions, *c.context.training))
        shared_ctx = [v for v, n in ctx.most_common(3) if n >= max(2, len(members) // 2)]
        subject = [e for e, _ in ents.most_common(1)]
        kw = [k for k in keywords if not any(s.lower() in k or k in s.lower() for s in subject)]
        parts = subject + (shared_ctx[:2] or kw[:2])
        seen: set[str] = set()
        uniq = [p for p in parts if not (p.lower() in seen or seen.add(p.lower()))]
        return " · ".join(uniq) or "misc"

    @staticmethod
    def _theme_summary(members: list[Claim], paper_ids: list[str], support: int, conflicts: list[ConflictAnalysis]) -> str:
        parts = [f"{len(members)} claim{'s' if len(members) != 1 else ''} from {len(paper_ids)} paper{'s' if len(paper_ids) != 1 else ''}"]
        if support:
            parts.append(f"{support} supporting/extending link{'s' if support != 1 else ''}")
        if conflicts:
            verdicts = Counter(c.verdict for c in conflicts)
            bits = []
            if verdicts[ConflictVerdict.CONTEXT_DEPENDENT]:
                bits.append(f"{verdicts[ConflictVerdict.CONTEXT_DEPENDENT]} explained by differing context")
            if verdicts[ConflictVerdict.GENUINE]:
                bits.append(f"{verdicts[ConflictVerdict.GENUINE]} under comparable conditions")
            if verdicts[ConflictVerdict.INSUFFICIENT_EVIDENCE]:
                bits.append(f"{verdicts[ConflictVerdict.INSUFFICIENT_EVIDENCE]} with insufficient context")
            parts.append(f"{len(conflicts)} apparent conflict{'s' if len(conflicts) != 1 else ''} ({'; '.join(bits)})")
        scales = Counter(c.context.data_scale for c in members if c.context.data_scale)
        if len(scales) > 1:
            parts.append("evidence spans " + ", ".join(f"{s} ({n})" for s, n in scales.most_common()) + " data regimes")
        return "; ".join(parts) + "."

    # ------------------------------------------------------------------ gaps
    def gaps(self, claims: list[Claim], papers: list[Paper], themes: list[Theme], conflicts: list[ConflictAnalysis]) -> list[ResearchGap]:
        gaps: list[ResearchGap] = []
        n_papers = len({c.paper_id for c in claims}) or 1
        by_id = {c.claim_id: c for c in claims}

        # 1. repeated limitations across papers
        limits = [c for c in claims if c.claim_type == ClaimType.LIMITATION]
        if len(limits) >= 2:
            vecs = self.models.embedder.embed([c.normalized_text for c in limits])
            used: set[int] = set()
            for i in range(len(limits)):
                if i in used:
                    continue
                group = [i] + [j for j in range(i + 1, len(limits)) if j not in used and float(vecs[i] @ vecs[j]) >= self.theme_threshold - 0.05]
                papers_in = {limits[k].paper_id for k in group}
                if len(papers_in) >= 2:
                    used.update(group)
                    rep = limits[group[0]]
                    gaps.append(ResearchGap(
                        gap_id="gap_" + _gid("lim", *[limits[k].claim_id for k in group]),
                        title=f"Potential research gap: recurring limitation — {clip(rep.claim_text, 140)}",
                        signal="repeated_limitation",
                        rationale=f"{len(papers_in)} of {n_papers} analysed papers report a similar limitation, suggesting it is not yet solved.",
                        evidence_claim_ids=[limits[k].claim_id for k in group],
                        strength="strong" if len(papers_in) >= 3 else "moderate",
                    ))

        # 2. unresolved conflicts
        for cf in conflicts:
            if cf.verdict in (ConflictVerdict.GENUINE, ConflictVerdict.INSUFFICIENT_EVIDENCE):
                a, b = by_id[cf.claim_a_id], by_id[cf.claim_b_id]
                gaps.append(ResearchGap(
                    gap_id="gap_" + _gid("cf", cf.conflict_id),
                    title="Potential research gap: unresolved disagreement between "
                          f"{a.source_citation} and {b.source_citation}",
                    signal="unresolved_conflict",
                    rationale=("Opposing findings under comparable conditions; no analysed study reconciles them."
                               if cf.verdict == ConflictVerdict.GENUINE else
                               f"The claims conflict but key context ({', '.join(cf.unknown_factors[:3])}) is unreported, "
                               "so a controlled comparison is missing from the analysed set."),
                    evidence_claim_ids=[a.claim_id, b.claim_id],
                    strength="moderate" if cf.verdict == ConflictVerdict.GENUINE else "weak",
                ))

        # 3. context-dependent conflicts → missing controlled comparison
        ctx_conflicts = [cf for cf in conflicts if cf.verdict == ConflictVerdict.CONTEXT_DEPENDENT]
        factor_counts = Counter(f.factor for cf in ctx_conflicts for f in cf.differing_factors[:2])
        for factor, count in factor_counts.most_common(2):
            ev = sorted({cid for cf in ctx_conflicts if any(f.factor == factor for f in cf.differing_factors[:2])
                         for cid in (cf.claim_a_id, cf.claim_b_id)})
            if count >= 2:
                gaps.append(ResearchGap(
                    gap_id="gap_" + _gid("factor", factor),
                    title=f"Potential research gap: few controlled comparisons across {factor}",
                    signal="missing_comparison",
                    rationale=(f"{count} apparent conflicts in the analysed set are explained by differing {factor}; "
                               f"a study varying {factor} while holding the rest fixed would test the dependence directly."),
                    evidence_claim_ids=ev[:8],
                    strength="moderate" if count >= 3 else "weak",
                ))

        # 4. under-studied conditions (data regimes / populations / datasets examined by one paper only)
        understudied: list[ResearchGap] = []
        for attr, label in (("data_scale", "data regime"), ("population", "population"), ("datasets", "dataset")):
            coverage: dict[str, set[str]] = {}
            evidence: dict[str, list[str]] = {}
            for c in claims:
                if c.claim_type in (ClaimType.BACKGROUND, ClaimType.METHOD):
                    continue
                values = getattr(c.context, attr)
                for v in values if isinstance(values, list) else ([values] if values else []):
                    coverage.setdefault(v, set()).add(c.paper_id)
                    evidence.setdefault(v, []).append(c.claim_id)
            if len(coverage) < 2:
                continue
            common = max(len(p) for p in coverage.values())
            for value, ps in coverage.items():
                if len(ps) == 1 and common >= 3:
                    understudied.append(ResearchGap(
                        gap_id="gap_" + _gid(attr, value),
                        title=f"Potential research gap: {label} '{value}' is examined by only one analysed paper",
                        signal="understudied_condition",
                        rationale=f"Other {label}s are covered by up to {common} papers; findings for '{value}' rest on a single source.",
                        evidence_claim_ids=evidence[value][:4],
                        strength="weak",
                    ))

        gaps.extend(understudied[:3])

        # 5. future work statements
        future = [c for c in claims if c.claim_type == ClaimType.FUTURE_WORK]
        for c in future[:4]:
            gaps.append(ResearchGap(
                gap_id="gap_" + _gid("fw", c.claim_id),
                title=f"Potential research gap (author-stated): {clip(c.claim_text, 140)}",
                signal="future_work",
                rationale=f"Explicitly named as future work / open question by {c.source_citation}.",
                evidence_claim_ids=[c.claim_id],
                strength="weak",
            ))

        # 6. relevant single-source themes
        for t in themes:
            if t.status == "single_source" and len(t.claim_ids) >= 2 and len(gaps) < 14:
                gaps.append(ResearchGap(
                    gap_id="gap_" + _gid("ss", t.theme_id),
                    title=f"Potential research gap: '{t.label}' is supported by a single paper",
                    signal="single_source",
                    rationale="No other analysed paper replicates or tests these claims.",
                    evidence_claim_ids=t.claim_ids[:4],
                    strength="weak",
                ))
        order = {"strong": 0, "moderate": 1, "weak": 2}
        gaps.sort(key=lambda g: order[g.strength])
        return gaps

    # ------------------------------------------------------------------ summary
    def build(self, question: str, papers: list[Paper], claims: list[Claim], relationships: list[Relationship],
              conflicts: list[ConflictAnalysis]) -> Landscape:
        themes = self.themes(claims, relationships, conflicts)
        gaps = self.gaps(claims, papers, themes, conflicts)
        consensus = [t for t in themes if t.status == "consensus"]
        disputed = [t for t in themes if t.status == "disputed"]
        # single-claim themes are noise at landscape level; they remain visible in the claims view
        emerging = [t for t in themes if t.status == "emerging" or (t.status == "single_source" and len(t.claim_ids) >= 2)]
        by_id = {c.claim_id: c for c in claims}
        unresolved = []
        for cf in conflicts:
            if cf.verdict != ConflictVerdict.CONTEXT_DEPENDENT:
                a, b = by_id[cf.claim_a_id], by_id[cf.claim_b_id]
                unresolved.append(f"Does “{clip(a.claim_text, 110)}” ({a.source_citation}) still hold given "
                                  f"“{clip(b.claim_text, 110)}” ({b.source_citation})?")
        for cf in conflicts:
            if cf.verdict == ConflictVerdict.CONTEXT_DEPENDENT and cf.differing_factors and len(unresolved) < 6:
                f = cf.differing_factors[0]
                a = by_id[cf.claim_a_id]
                unresolved.append(f"How does the finding “{clip(a.claim_text, 100)}” change across {f.factor} "
                                  f"({', '.join(f.a) or '?'} vs {', '.join(f.b) or '?'})?")
        summary = self._summary(question, papers, claims, relationships, conflicts, consensus, disputed)
        return Landscape(summary=summary, consensus=consensus, disagreements=disputed, emerging=emerging,
                         unresolved_questions=unresolved[:6], gaps=gaps)

    def _summary(self, question: str, papers: list[Paper], claims: list[Claim], relationships: list[Relationship],
                 conflicts: list[ConflictAnalysis], consensus: list[Theme], disputed: list[Theme]) -> str:
        years = sorted(p.year for p in papers if p.year)
        span = f"{years[0]}–{years[-1]}" if years else "unknown years"
        rel_counts = Counter(r.relation.value for r in relationships)
        verdicts = Counter(c.verdict for c in conflicts)
        full = sum(1 for p in papers if p.text_source == "full_text")
        text = (f"ClaimGraph analysed {len(papers)} papers ({span}; {full} with full text, {len(papers) - full} abstract-only) "
                f"and extracted {len(claims)} claims linked by {len(relationships)} cross-paper relationships "
                f"({', '.join(f'{v} {k.lower()}' for k, v in rel_counts.most_common()) or 'none'}). ")
        if consensus:
            text += f"Consensus themes: {', '.join(t.label for t in consensus[:3])}. "
        if disputed:
            text += f"Disputed themes: {', '.join(t.label for t in disputed[:3])}. "
        if conflicts:
            ctx, gen, ins = (verdicts[v] for v in (ConflictVerdict.CONTEXT_DEPENDENT, ConflictVerdict.GENUINE,
                                                   ConflictVerdict.INSUFFICIENT_EVIDENCE))
            text += (f"Of {len(conflicts)} apparent disagreement{'s' if len(conflicts) != 1 else ''}, {ctx} "
                     f"{'is' if ctx == 1 else 'are'} plausibly explained by differing context, {gen} "
                     f"{'occurs' if gen == 1 else 'occur'} under comparable conditions, and {ins} "
                     f"{'lacks' if ins == 1 else 'lack'} enough reported context to judge.")
        if self.llm.enabled and self.llm.remaining > 0 and claims:
            top = sorted(claims, key=lambda c: -c.relevance)[:8]
            facts = "\n".join(f"- {c.source_citation}: {c.claim_text}" for c in top)
            out = self.llm.complete_json(
                "Write a 3-sentence neutral synthesis answering the research question using ONLY the listed claims and "
                "statistics. Cite papers as given. Do not add facts.",
                f"Question: {question}\nStatistics: {text}\nClaims:\n{facts}",
                {"type": "object", "properties": {"synthesis": {"type": "string"}}, "required": ["synthesis"]},
                max_tokens=220,
            )
            if out and out.get("synthesis"):
                text += "\n\nLLM synthesis (grounded in the claims above): " + str(out["synthesis"]).strip()
        return text.strip()
