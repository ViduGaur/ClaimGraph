# ClaimGraph

**An AI system for cross-paper scientific claim analysis.**
Give it a research question. ClaimGraph:
1. discovers relevant open-access papers
2. extracts their claims together with verbatim evidence
3. compares the claims across papers
4. classifies how they relate: **SUPPORTS**, **CONTRADICTS**, **QUALIFIES** or **EXTENDS**
5. explains *why* apparently conflicting findings disagree (dataset, data scale, conditions, training, metric, population, …)
6. summarises the result as a claim graph and a research landscape: consensus, disagreements, unresolved questions and *potential* research gaps

It is a multi-document reasoning pipeline, not a chatbot and not "summarise these PDFs with an LLM". Every claim is traceable to the sentence it came from, and every relationship shows the signals and rules that produced it.

The core runs **free and locally**: public scholarly APIs, small ONNX models on the CPU, NetworkX and SQLite. An LLM is optional; with one (OpenRouter by default) ClaimGraph runs as a **hybrid**: the LLM reads the passages and fills in a quote-backed comparison, and the same transparent rules make the decision.

There is also an **evaluation harness** that compares ClaimGraph with baselines and ablations on held-out topics against blinded, model-assisted reference labels: see [`eval/README.md`](eval/README.md).

---

## Contents
[Architecture](#architecture) · [Pipeline](#pipeline) · [Tech stack](#tech-stack) · [Setup](#setup) · [Running](#running) · [Terminal](#from-the-terminal) · [Demo mode](#demo-mode) · [Configuration](#environment-variables) · [LLM](#optional-llm-hybrid-mode) · [Evaluation](#evaluation) · [API](#api) · [Tests](#tests) · [Limitations](#known-limitations) · [Future work](#future-improvements) · [Layout](#repository-layout)

More detail: [`docs/architecture.md`](docs/architecture.md) (design and decision rules) and [`docs/research.md`](docs/research.md) (provider and model research, with pricing and limits checked Sep 2026).

---

## Architecture

```
frontend/ (React + Vite + Cytoscape.js)  ──HTTP──▶  backend/ (FastAPI)
                                                     │
      POST /research/analyze ─▶ AnalysisService ─▶ background worker ─▶ AnalysisPipeline
                                  │  (SQLite: status, progress, result JSON)          │
                                  ▼                                                   ▼
                              GET /research/{id}/…            literature/  extraction/  reasoning/  graph/
                                                               providers/ (LLM interface, local ONNX models)
```

- **Backend:** FastAPI + Pydantic. Route handlers only call `AnalysisService`, and all logic lives in the pipeline modules.
- **Persistence:** SQLite through SQLAlchemy, a single `analyses` table holding status, stage progress and the result document. Postgres would add operational weight without changing anything the app needs.
- **Models:** interchangeable providers.
  - `LLMProvider` → `OpenRouterProvider` | `OllamaProvider` | `HuggingFaceProvider` | `OpenAICompatibleProvider` (optional), behind a call budget and a disk cache.
  - `Embedder`, `Reranker`, `NLIModel`: local ONNX models, each with a dependency-free fallback.
- **Graph:** NetworkX for construction and analytics (centrality, bridging papers, components). Cytoscape.js renders it in the browser.

## Pipeline

| # | Stage | How |
|---|---|---|
| 1 | Query understanding | Rule-based key-term and entity extraction (LLM query expansion if enabled); biomedical questions automatically add Europe PMC |
| 2 | Literature discovery | OpenAlex + Semantic Scholar + arXiv (+ Europe PMC), queried in parallel with per-host rate limiting and retries |
| 3 | Paper ranking | Deduplication (DOI → arXiv id → fuzzy title), then cross-encoder relevance (`ms-marco-MiniLM-L-6-v2`) plus priors for open access, citations and abstract availability |
| 4 | Open-access acquisition | OA PDF URLs reported by the metadata APIs or arXiv → Europe PMC full-text XML (OA subset, resolved by DOI) → Unpaywall; robots.txt honoured; size-capped; cached |
| 5 | PDF/text extraction | PyMuPDF with **column-aware reading order**, running-header and figure/table removal, section detection (abstract/methods/results/…), references dropped; JATS XML parsed into sections directly |
| 6 | Relevant passage retrieval | Sections chunked into passages, embedded (`bge-small-en-v1.5`) and ranked against the question; only the abstract plus the top passages are mined |
| 7 | Claim extraction | Transparent cue-based sentence scoring (reporting verbs, comparatives, quantitative results, hedges, limitation and future-work markers) × section weight × relevance. With an LLM, the model **screens the verbatim candidates** (keep or drop, never rewrite) |
| 8 | Claim normalisation | Alias canonicalisation (ConvNets→CNN, Vision Transformer→ViT, …), reporting-frame stripping, within-paper deduplication; claim type, polarity, subject/object, reported result, `ClaimContext` |
| 9 | Claim comparison | Cross-paper candidate pairs by embedding similarity and shared entities |
| 10 | Relationship classification | Bidirectional NLI (`nli-deberta-v3-xsmall`) + stance + context comparison → explicit decision rules. Stance and the context table come from surface rules, or, with an LLM, from a **grounded pair analysis** in which every context value needs a verbatim quote |
| 11 | Contextual disagreement analysis | Factor-by-factor comparison (12 factors) → verdict: *context-dependent*, *genuine disagreement* or *insufficient evidence*, with a structured explanation |
| 12 | Claim graph | NetworkX MultiDiGraph (paper → claim → relation → claim → paper) + analytics |
| 13 | Research landscape | Claim communities (modularity) → consensus / disputed / emerging themes; hedged "Potential research gap" signals, each with its evidence |

**Key rule:** ClaimGraph never forces a contradiction. Its output has two levels: a **relation** (SUPPORTS, EXTENDS, QUALIFIES, CONTRADICTS) and, for pairs that conflict at face value, a **conflict verdict** (genuine, context-dependent, insufficient evidence).
- Opposing claims made under *different* stated conditions become **QUALIFIES**, verdict *context-dependent*. QUALIFIES means one claim narrows, conditions or contextualises the other; it does not mean they agree.
- Opposing claims whose passages state *no* comparable context keep the relation CONTRADICTS but are flagged uncertain and **not asserted**, verdict *insufficient evidence*.
- Only opposing claims under *comparable* stated conditions become an asserted **CONTRADICTS**, verdict *genuine*.
- Every verdict comes with an explanation record: the apparent conflict, each decisive mismatch with both papers' values and verbatim quotes, and a short grounded explanation.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI, Pydantic v2, Uvicorn | typed schemas, OpenAPI docs at `/api/docs` |
| Storage | SQLite (SQLAlchemy 2) | zero-ops; one JSON result document per analysis |
| Discovery | OpenAlex, Semantic Scholar, arXiv, Europe PMC, Unpaywall | free, legal, keyless (keys optional) |
| PDFs | PyMuPDF | fast, good text fidelity; column-aware ordering added |
| Embeddings / reranking | fastembed (ONNX): `BAAI/bge-small-en-v1.5`, `Xenova/ms-marco-MiniLM-L-6-v2` | CPU-fast, no PyTorch |
| NLI | `cross-encoder/nli-deberta-v3-xsmall` (ONNX, onnxruntime) | contradiction/entailment signal without an LLM |
| LLM (optional) | OpenRouter (default) / Ollama / Hugging Face router / any OpenAI-compatible endpoint | behind one interface, hard call budget |
| Graph | NetworkX + Cytoscape.js | analytics + lightweight interactive rendering |
| Frontend | React 19, Vite, TypeScript, self-hosted IBM Plex + Source Serif fonts | no CSS framework, no external calls |

## Setup

Requirements: Python 3.11+ (tested with 3.12), Node 20+ (tested with 22), about 500 MB of disk for the local models.

```bash
git clone https://github.com/ViduGaur/ClaimGraph.git && cd ClaimGraph

# backend — an isolated venv inside backend/
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt      # runtime deps + pytest
cd ..

# frontend
cd frontend && npm install && cd ..

# optional configuration (everything works without it)
cp .env.example .env
```

The first analysis downloads the three ONNX models (about 430 MB) from Hugging Face into `backend/data/models/`. No token or account is needed, and they are cached after that.

## Running

Two terminals:

```bash
# terminal 1 — backend on http://127.0.0.1:8765  (API docs: /api/docs)
cd backend && source .venv/bin/activate
uvicorn app.main:app --port 8765 --reload

# terminal 2 — frontend on http://localhost:5199 (proxies /api to :8765)
cd frontend && npm run dev
```

Or run both in the background with `scripts/dev.sh start`, and stop them with `scripts/dev.sh stop`. Logs go to `logs/`.

The defaults are backend port 8765 and frontend port 5199. To change them, set `CLAIMGRAPH_BACKEND_PORT` / `CLAIMGRAPH_FRONTEND_PORT` for `scripts/dev.sh` (or `CLAIMGRAPH_FRONTEND_PORT` and `CLAIMGRAPH_API` for `npm run dev`) and add the frontend origin to `CLAIMGRAPH_CORS_ORIGINS`.

## From the terminal

The whole analysis is available without the UI. `cg_run.py` prints the extracted claims with their contexts, every related claim pair with its similarity, NLI scores and context-factor table, the relation, the conflict verdict and the evidence behind it:

```bash
backend/.venv/bin/python scripts/cg_run.py "Does large-batch training hurt generalization?" --max-papers 8
backend/.venv/bin/python scripts/cg_run.py --demo --no-llm          # bundled corpus, rule-based pipeline
backend/.venv/bin/python scripts/cg_run.py --demo --json out.json   # also save the full result document
```

## Demo mode

Demo mode runs the **identical pipeline** on a bundled corpus of 10 real papers: arXiv metadata and abstracts, which arXiv releases under CC0 (`backend/app/demo/corpus.json`, rebuilt with `scripts/build_demo_corpus.py`). Only the literature source is swapped and network downloads are skipped. Nothing in it is synthetic.

The topic is *"Do vision transformers need large-scale pre-training to outperform convolutional networks?"*. It was chosen because the literature has well-known apparent contradictions that are resolved by context (data scale, augmentation/regularisation, training recipe).

**First end-to-end analysis:**
1. Start the backend and frontend (above).
2. Open http://localhost:5199, tick **Demo corpus**, click the first example question, then **Analyze literature**.
3. The first run takes about 1–2 min while the models download; after that about 30–40 s on a laptop CPU.
4. Explore the tabs: Claim graph, Claims (click any claim → *Why did ClaimGraph say this?*), Disagreements, Landscape and Papers.

Alternatives:
- `CLAIMGRAPH_DEMO_MODE=true` makes demo the default for every run.
- From the command line: `backend/.venv/bin/python scripts/cg_run.py --demo`.
- Through the API:

```bash
curl -X POST localhost:8765/api/v1/research/analyze -H 'content-type: application/json' \
  -d '{"question":"Do vision transformers need large-scale pre-training to outperform convolutional networks?","demo":true}'
```

**Live mode** is the default: untick *Demo corpus* and ask anything. ClaimGraph searches OpenAlex, Semantic Scholar and arXiv, downloads open-access PDFs (arXiv at one request every 3 s), and analyses the full text. Six to eight papers take about 1–4 minutes.

## Environment variables

All are optional and all are prefixed `CLAIMGRAPH_`, so generic names such as `DATABASE_URL` exported for other projects are never picked up. See [`.env.example`](.env.example).

| Variable | Default | Meaning |
|---|---|---|
| `CLAIMGRAPH_DEMO_MODE` | `false` | Default to the bundled demo corpus |
| `CLAIMGRAPH_CONTACT_EMAIL` | – | Polite-pool contact for OpenAlex/arXiv; **required for Unpaywall** OA lookups |
| `CLAIMGRAPH_OPENALEX_API_KEY` | – | Optional free key: $1/day budget instead of $0.10/day (≈1,000 vs ≈100 searches) |
| `CLAIMGRAPH_SEMANTIC_SCHOLAR_API_KEY` | – | Optional free key; the keyless pool is shared and often returns 429 (ClaimGraph then continues without it) |
| `CLAIMGRAPH_SOURCES` | `openalex,semantic_scholar,arxiv` | Add `europepmc` to always include it |
| `CLAIMGRAPH_MAX_PAPERS` | `8` | Papers analysed per question (UI can override, 2–20) |
| `CLAIMGRAPH_DOWNLOAD_PDFS` | `true` | `false` = abstracts only (faster) |
| `CLAIMGRAPH_LLM_PROVIDER` | `auto` | `auto` \| `none` \| `openrouter` \| `ollama` \| `huggingface` \| `openai_compatible`. `auto` = OpenRouter when a key is set, otherwise no LLM |
| `CLAIMGRAPH_LLM_API_KEY` | – | OpenRouter key (also the key for `openai_compatible` / `huggingface`) |
| `CLAIMGRAPH_LLM_MODEL` | provider default | `inclusionai/ling-3.0-flash` on OpenRouter, `qwen2.5:7b` on Ollama |
| `CLAIMGRAPH_LLM_ROUTE` | `Novita` | OpenRouter upstream host to prefer (hosts differ in speed and quantisation) |
| `CLAIMGRAPH_LLM_MAX_CALLS` | `0` | Hard LLM budget per analysis; `0` = 80 for hosted providers, 6 for Ollama |
| `CLAIMGRAPH_LLM_CLAIM_SELECTION` | `true` | LLM screens each paper's verbatim claim candidates |
| `CLAIMGRAPH_LLM_PAIR_ANALYSIS` | `true` | LLM stance + quote-backed context table per claim pair |
| `CLAIMGRAPH_LLM_CACHE` | `true` | Replay identical LLM calls from `backend/data/llm_cache.jsonl` |
| `CLAIMGRAPH_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `CLAIMGRAPH_LLM_API_BASE_URL` | – | For `openai_compatible` (Groq, vLLM, LM Studio, …) |
| `CLAIMGRAPH_EMBEDDING_BACKEND` | `fastembed` | `hashing` = no model download (much weaker) |
| `CLAIMGRAPH_NLI_BACKEND` | `onnx` | `lexical` = no model download (much weaker) |
| `CLAIMGRAPH_CORS_ORIGINS` | `http://localhost:5199,http://127.0.0.1:5199` | Comma-separated origins allowed to call the API |
| `CLAIMGRAPH_DATABASE_URL` | `sqlite:///backend/data/claimgraph.db` | Any SQLAlchemy URL |

### API keys you may want (none are required)

| Key | Where | Cost | Effect |
|---|---|---|---|
| OpenAlex | https://openalex.org/settings/api (free account) | free | 10× daily search budget |
| Semantic Scholar | https://www.semanticscholar.org/product/api#api-key-form | free | reliable 1 req/s instead of the contended shared pool |
| Hugging Face token | https://huggingface.co/settings/tokens | free tier ≈ $0.10/month of router credit | only if `LLM_PROVIDER=huggingface` |

## Optional: LLM (hybrid mode)

The whole pipeline works without an LLM. Put an OpenRouter key in `.env` and it switches to hybrid mode:

```bash
echo 'CLAIMGRAPH_LLM_API_KEY=sk-or-...' >> .env     # default model: inclusionai/ling-3.0-flash
```

What the LLM does, and what it is not allowed to do:

| Step | The LLM | The guard |
|---|---|---|
| Claim selection | picks, from the numbered cue-based candidate sentences of a paper, those that are real findings | it returns sentence numbers only, so every claim is still a verbatim sentence |
| Pair analysis | names each claim's subject and outcome, says whether they are the same relationship, restates both conclusions and labels the stance; fills a 12-factor context table | every context value needs a verbatim quote that is checked against the paper; a factor stated by one side only is *unknown*, never *different* |
| Decision | nothing | the relation and the conflict verdict come from the same rules as in the rule-based pipeline ([`docs/context-gate.md`](docs/context-gate.md) documents them exactly) |
| Explanation | words each conflict verdict after it has been made | it sees only the decision's own evidence record; its text is replaced by a deterministic template if it cites a factor, value or number that is not in that record |
| Summary | writes a short summary | it may only use the extracted claims |

Requests use temperature 0 and a fixed seed, and every response is cached on disk, so repeating an analysis replays the same answers. Calls run under a hard budget (`CLAIMGRAPH_LLM_MAX_CALLS`); a failed, malformed or over-budget call falls back to the rule-based result for that paper or pair.

Other providers use the same interface: `CLAIMGRAPH_LLM_PROVIDER=ollama` for a local model (about one minute per call on a CPU, so the default budget drops to 6), `openai_compatible` with `CLAIMGRAPH_LLM_API_BASE_URL`, or `huggingface`.

## Evaluation

`scripts/cg_eval.py` runs the experiments: frozen topic corpora, claim pairs with all their signals, baselines (similarity, NLI, direct LLM classification), ablations (without the context gate, without stance, without NLI, without quote grounding, without the LLM), degenerate gates, blinded model-assisted annotation with researcher adjudication, metrics with bootstrap confidence intervals, an analysis of why apparent contradictions were rejected, and an audit of the explanations. See [`eval/README.md`](eval/README.md), which also explains how to check the freeze record (`python eval/verify_freeze.py`). A manuscript is in preparation and is not part of this repository.

## API

Interactive docs: http://127.0.0.1:8765/api/docs

| Method | Route | Returns |
|---|---|---|
| `POST` | `/api/v1/research/analyze` | `202` + analysis summary (runs in background) |
| `GET` | `/api/v1/research` | recent analyses |
| `GET` | `/api/v1/research/{id}` | status, stage progress, warnings, full result (`?include_result=false` to poll) |
| `GET` | `/api/v1/research/{id}/papers` | ranked papers with OA status and text source |
| `GET` | `/api/v1/research/{id}/claims` | claims (`?paper_id=`, `?claim_type=`) |
| `GET` | `/api/v1/research/{id}/claims/{claim_id}` | "why did ClaimGraph say this": claim, paper, relationships, related claims |
| `GET` | `/api/v1/research/{id}/relationships` | typed relations with signals, factors and reasoning |
| `GET` | `/api/v1/research/{id}/graph` | nodes, edges and graph statistics |
| `GET` | `/api/v1/research/{id}/conflicts` | contextual conflict analyses |
| `GET` | `/api/v1/research/{id}/gaps` | potential research gaps and the landscape |
| `GET` | `/api/v1/health` | mode, sources, LLM availability, which optional keys are set |

## Tests

```bash
cd backend && source .venv/bin/activate && pytest
```

The 158 tests run offline with the dependency-free fallback models and never call an LLM; one test exercises the real ONNX NLI model and is skipped unless it is cached. They cover:
- **Literature retrieval:** parsing, malformed records, 429 retries, failing sources
- **Deduplication**
- **Claim schema validation**
- **Extraction:** sentences, sections, context, claim typing, polarity, grounding
- **Relationship classification:** each verdict path, with a pinned NLI
- **Graph construction**
- **Landscape and gaps**
- **Hybrid mode:** quote grounding (ungrounded and one-sided context is never "different"), LLM claim selection, fallback to rules on failure, the response cache
- **Ablation switches and the evaluation harness:** label derivation, agreement, metrics, systems as pure functions of a pair record
- **Failure handling:** LLM budget and failures, invalid PDFs, missing OA copies, partial extraction failures
- **API endpoints**
- **End to end** on the demo corpus

## Known limitations

- **Claim candidates are heuristic.** Cue-based sentence scoring is transparent and fast but misses implicit claims; without an LLM it also keeps some descriptive sentences. With an LLM the candidates are screened, but a claim the cues never proposed cannot be recovered.
- **Without an LLM, context extraction is gazetteer/regex-based.** It knows many ML/vision/NLP datasets, metrics and model families and some biomedical cues. Other fields get less structured context, so more of their conflicts land in *insufficient evidence*. Hybrid mode replaces the lexicon with a quote-backed table, at the price of depending on the LLM's reading of the passages.
- **The LLM's stance judgement is the weakest link of hybrid mode.** Deciding whether two claims are about the same relationship is hard for a small model; the grounding checks constrain what it may say about context, not whether it relates the right pairs.
- **NLI models are brittle on long, multi-clause scientific sentences.** For example, "method X improves Y" vs "method Y improves Y" often scores as a contradiction. Stance rules block the common cases, but some relationships are still wrong. Uncertain ones are flagged.
- **PDF parsing** handles one- and two-column layouts; complex layouts, tables, equations and scanned PDFs are degraded or skipped. GROBID would be the next step.
- **Only open-access text is used.** Paywalled papers contribute only their abstract, or are skipped. Some OA publishers (e.g. Wiley, Hindawi, MDPI, ScienceDirect) refuse automated downloads or serve HTML; ClaimGraph respects that and falls back to Europe PMC or the abstract.
- **Small paper sets.** 6–12 papers per question; gap detection is relative to *that* set, never the whole literature.
- **Fallback models** (`hashing`/`lexical`) keep the app runnable offline but find far fewer relationships.
- **Single worker.** Analyses run one at a time; that's fine for a local tool, not a multi-user service.

## Future improvements

- GROBID or Nougat for structured full text (sections, tables, citations)
- A trained scientific claim extractor (e.g. fine-tuned on SciFact / SciClaim) replacing the cue rules
- Scientific NLI (e.g. SciNLI / MedNLI-tuned cross-encoders) and numeric result comparison (same metric and dataset → compare the actual numbers)
- Citation-graph signals (does paper B cite and dispute paper A?)
- Manual PDF upload as an extra input
- Export (JSON-LD / GraphML / Markdown report)

## Repository layout

| Path | Contents |
|---|---|
| `backend/app/` | the FastAPI application and the analysis pipeline (`literature`, `extraction`, `reasoning`, `graph`, `providers`, ...), and `evaluation/`, the experiment harness, annotation protocols and studies |
| `backend/tests/` | offline test suite |
| `frontend/` | React + Vite + TypeScript UI |
| `eval/` | evaluation data: topics, freeze record and its verifier, reference labels and raw annotator calls, stored claim frames, result summaries; start with [`eval/README.md`](eval/README.md) |
| `docs/` | architecture, decision-rule reference, provider research; index in [`docs/README.md`](docs/README.md) |
| `scripts/` | `dev.sh` (start/stop both servers), `cg_run.py` (pipeline from the terminal), `cg_eval.py` (evaluation CLI), `build_demo_corpus.py` |
| `.github/workflows/` | CI: backend tests, frontend lint and build |

### Where to find things

| You want | Look in |
|---|---|
| the implementation | `backend/app/` (design in [`docs/architecture.md`](docs/architecture.md)) |
| the experiments | `scripts/cg_eval.py` and `backend/app/evaluation/`, documented in [`eval/README.md`](eval/README.md) |
| the annotations | code in `backend/app/evaluation/{annotation,model_annotation,conflict_annotation}.py`, labels and raw annotator calls in `eval/annotation/` |
| the results | `eval/results/` |
| the pre-registration and freeze record | `eval/annotation/f1/PREREGISTRATION.md`, `eval/FREEZE.json`, `eval/verify_freeze.py` |

## Authors

Vidu Gaur, Vidip Gaur and Naman Daga.
