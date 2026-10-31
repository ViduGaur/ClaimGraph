# ClaimGraph — architecture

## 1. Goals that shaped the design

1. **Multi-document reasoning, not summarisation.** The unit of analysis is the *claim*, and claims from different papers are compared pairwise with explicit signals.
2. **Evidence grounding.** Every claim is a verbatim sentence (or an LLM claim that fuzzy-matches one) inside a stored passage with its section. Every relationship stores its signals, the factor comparison and a reasoning string.
3. **Calibrated honesty.** Uncertainty is represented (flags, confidence, "not stated", "from paper"). Contradictions are never forced, and gaps are always "potential".
4. **Free and local by default.** Public APIs, CPU ONNX models, SQLite. Paid or hosted services sit behind interfaces and are optional.
5. **Graceful degradation.** Any single source, paper, model or LLM call can fail without failing the analysis.

## 2. Layout

```
backend/app/
  main.py                FastAPI app factory + lifespan (DB init, worker, stale-run recovery)
  api/routes.py          thin HTTP layer → AnalysisService
  core/                  config.py (Settings: CLAIMGRAPH_* env vars, .env), logging.py
  db/session.py          SQLite model: analyses(id, status, stages, warnings, counts, result JSON)
  schemas/               domain.py (Paper, Passage, Claim, ClaimContext, Relationship, ConflictAnalysis, Theme,
                         ResearchGap, graph), api.py (request/response models, stage list)
  services/              analysis_service.py: background worker, progress persistence
  pipelines/analysis.py  the orchestrator: stage order, failure isolation, warnings
  literature/            http.py (polite client), sources.py (adapters), discovery.py (dedup + ranking)
  extraction/            acquisition.py (OA → PDF), text.py (PDF → sections → sentences → passages),
                         context.py (ClaimContext extraction), claims.py (claim detection/structuring)
  reasoning/             query.py, relations.py (decision rules), alignment.py (grounded LLM pair analysis → PairView),
                         conflicts.py (verdicts), explanations.py (grounded wording of a verdict), landscape.py,
                         comparability.py and frames.py (studied alternatives, not wired into the pipeline)
  graph/builder.py       NetworkX graph + statistics + serialisation
  providers/             models.py (Embedder / Reranker / NLIModel: ONNX + fallbacks),
                         llm/ (LLMProvider interface; OpenRouter, Ollama, Hugging Face, OpenAI-compatible;
                         BudgetedLLM; disk cache)
  evaluation/            experiment harness, annotation protocols and studies (map in eval/README.md)
  demo/corpus.json       real arXiv abstracts (CC0) for DEMO_MODE
backend/tests/           offline test suite (conftest.py holds the fixtures)
frontend/src/
  api/, hooks/, types.ts           typed client + polling hook (types mirror schemas/domain.py)
  pages/                           Home (question form), AnalysisPage (progress, tab bar)
  features/                        Workspace (paper rail, graph, detail panel), Claims, Disagreements,
                                   Landscape, Papers
  graph/ClaimGraphView.tsx         Cytoscape.js view
  components/                      ClaimDetail, RelationDetail, ClaimDrawer ("Why did ClaimGraph say this?"),
                                   UI primitives
  lib/paperColors.ts               per-paper palette
eval/                    topics, freeze record, reference labels, result summaries (see eval/README.md)
scripts/                 dev.sh, cg_run.py (pipeline from the terminal), cg_eval.py (evaluation CLI),
                         build_demo_corpus.py
```

## 3. Data model (abridged)

```
Claim
  claim_id, paper_id, claim_text (verbatim), normalized_text
  claim_type: finding | comparison | method | limitation | future_work | background
  polarity: positive | negative | neutral | none         subject (canonical entity)
  evidence: [Passage(section, text)]                      section
  context: ClaimContext(datasets, data_scale, conditions, training, models, baselines,
                        metrics, task, population, sample_size, methodology, time_period)
  inherited_context: fields filled from paper-level text (weaker)
  reported_result, confidence, extraction_method, grounding_score, source_citation, uncertainty_notes

Relationship
  source_claim_id (later) → target_claim_id (earlier), relation, confidence, uncertain, decided_by
  signals: similarity, nli_entailment, nli_contradiction, nli_neutral, polarity_conflict, context_divergence
  factors: [FactorComparison(factor, a, b, status: same|overlap|different|unknown, inherited)]
  reasoning

ConflictAnalysis
  verdict: context_dependent | genuine_disagreement | insufficient_evidence
  headline, explanation, differing_factors, shared_factors, unknown_factors, evidence_passage_ids, confidence
```

## 4. Relationship decision rules

For each candidate pair (different papers, similarity ≥ threshold, or shared entity and near the threshold):

```
con = mean(NLI contradiction A→B, B→A)      ent = max(NLI entailment A→B, B→A)
stance = compare (subject family, object family, polarity) of both claims
divergence = weighted share of known factors that differ (paper-level values count ½; unknown ignored)

contradiction signal = (con ≥ 0.7 and stance ≠ agree and not same direction of effect)
                       or (opposing stance and con ≥ 0.2)
                       or (opposing stance and ent < 0.2 and similarity high)
  ├─ no stated experimental context      → CONTRADICTS, uncertain   → verdict insufficient_evidence
  ├─ divergence ≥ 0.3 and a factor differs → QUALIFIES               → verdict context_dependent
  └─ otherwise                            → CONTRADICTS             → verdict genuine_disagreement
agreement = ent ≥ 0.5, or ent ≥ 0.3 with high similarity, or the same directional stance
            (shared comparison object, or high similarity)
  ├─ a dataset / scale / task / population / condition differs → EXTENDS
  └─ otherwise                                                → SUPPORTS
parity vs directional stance on the same comparison            → QUALIFIES (uncertain)
same subject, very similar, NLI-neutral, adds a condition      → EXTENDS (uncertain)
```

Why these rules:
- **Stance guards.** Off-the-shelf NLI models, applied to long scientific sentences, often call "X improves Y" vs "Z improves Y" a contradiction. The stance guards remove that failure mode.
- **Unknown ≠ different.** A factor stated by only one side is *unknown*, not *different*, so missing information can never manufacture an explanation.
- **Relative thresholds.** Similarity cut-offs are relative to the embedding backend's pairing threshold (bge-small scores sit higher than the hashing fallback).
- **Direction.** Edges run from the later paper to the earlier one, following the publication year.

### 4.1 Hybrid mode: the LLM reads, the rules decide

The rules above need two inputs per pair besides similarity and NLI: a *stance* (do the claims agree or oppose at face value?) and a *factor table* (which context values does each claim state?). `PairView` carries both. It has two producers, and the decision rules are identical for either:

| | `rule_view` (no LLM) | `LLMPairAnalyzer` (`reasoning/alignment.py`) |
|---|---|---|
| stance | canonical subject/object entities + polarity cues | the LLM names each claim's subject and outcome, says whether the relationship is the same, restates both conclusions, then labels `agree` / `oppose` / `partial` / `unrelated`, ignoring differences in setting |
| factor table | lexicon and regex matches, compared as sets | for each of the 12 factors, the value each claim was established under **with a verbatim quote**, a status and whether the difference is material |

The LLM's table is never trusted as evidence. Code enforces:
- a value whose quote is not found in that paper's text is dropped;
- a factor grounded on one side only is `unknown`, never `different`;
- a value grounded only in the abstract is marked paper-level;
- a stated difference judged incidental is kept as `overlap`, so it cannot by itself turn a conflict into QUALIFIES.

Differences to the rule path, all in `RelationClassifier.decide`:
- a semantic `oppose` needs no NLI corroboration, and `unrelated` ends the analysis even if NLI reports a contradiction;
- one material stated difference is enough for the context gate (lexicon matches must add up to divergence ≥ 0.3);
- paper-level values count as known context, because the model reports a value only where it judged that it applies to the claim.

The LLM is also used as a **claim selector** (`ClaimExtractor._select_with_llm`): it sees the numbered cue-based candidate sentences of a paper and returns the numbers of those that are real findings. It can keep or drop a sentence, never rewrite one, so claims stay verbatim by construction.

Every LLM call goes through `BudgetedLLM`; a failure, a malformed answer or an exhausted budget leaves the rule-based result in place. Responses are cached on disk (`providers/llm/cache.py`), so a repeated run replays them.

### 4.2 Explanations

After the verdicts are final, `reasoning/explanations.py` attaches to every face-value conflict an explanation record
built from the decision's own factor table (apparent conflict, relation, verdict, each decisive mismatch with both
papers' values and quotes) and asks the LLM to word it. The wording is replaced by the deterministic template when it
cites a factor, value or number that is not in the record. The exact gate logic, including its quirks, is in
[`context-gate.md`](context-gate.md).

### 4.3 Ablation switches

`RuleConfig(use_stance, use_context_gate)` turns off the two components of the decision procedure:
- `use_context_gate=False`: every disagreement signal becomes CONTRADICTS (no QUALIFIES downgrade, no abstention);
- `use_stance=False`: no stance signal and no stance guards, so only NLI can raise a conflict or an agreement.

The evaluation harness (`app/evaluation`, see `eval/README.md`) builds every compared system from these switches and the two `PairView` producers, as pure functions over stored pair records.

## 5. Landscape and gap signals

- **Themes:** greedy-modularity communities on a graph of claims, connected by similarity ≥ 0.8 (bge) and by the classified relationships.
  - *consensus*: ≥ 2 papers, supporting or extending links, no conflicts
  - *disputed*: contains an apparent conflict
  - *emerging*: ≥ 2 papers, no links
  - *single-source*: one paper only
  - Labels combine the dominant subject with the most shared context values.
- **Potential research gaps**, always scoped to the analysed set and backed by claim IDs:
  - repeated limitations across papers
  - unresolved conflicts (genuine, or insufficient context)
  - the factors that most often explain context-dependent conflicts ("few controlled comparisons across data scale")
  - data regimes, populations or datasets covered by only one paper
  - author-stated future work
  - single-source themes

## 6. Failure handling

| Failure | Behaviour |
|---|---|
| Source returns 429/5xx | retried with backoff (honours `Retry-After`); after that the source is skipped and a warning is recorded |
| Malformed metadata | the record is skipped; other records are kept |
| Duplicate papers | merged (DOI, arXiv id, fuzzy title ≥ 93 and year ±1), keeping the best abstract and PDF link |
| No OA copy / HTML paywall / invalid or encrypted / scanned PDF | tries Europe PMC full-text XML, then falls back to the abstract, with a note on the paper |
| Extraction crash for one paper | that paper is skipped with a warning |
| Duplicate claims | within-paper near-duplicates (cosine ≥ 0.93) are merged |
| LLM unreachable / timeout / bad JSON / budget exhausted | `BudgetedLLM` returns `None` and the rule-based result stands (cue-ranked claims, `rule_view` for that pair) |
| LLM quotes text that is not in the paper | that context value is dropped; the factor becomes `unknown` |
| Models cannot be downloaded | hashing embeddings + lexical NLI, with a warning in the run |
| < 2 analysable papers or no claims | `PipelineError` with a user-facing message; the run is marked failed |
| Server restart mid-run | the run is marked failed on next start-up |

## 7. Visual encoding

Relationship colours were checked with a palette validator for colour-vision deficiency. Light mode uses aqua, orange, violet and blue; dark mode uses the same hues stepped for the dark surface, with magenta for *qualifies*. Colour is never the only cue:
- supports = solid line
- contradicts = heavier solid line
- qualifies = dashed
- extends = dotted
- uncertain = faded

Labels and a legend accompany every use. The claim graph uses a deterministic "by paper" layout: papers on an outer ring ordered by year, each paper's claims fanned inward, so edges read *paper → claim → relation → claim → paper*. A force layout is available as a toggle.
