# Provider and model research (September 2026)

For each external service: is it free, does it need a key, what are the limits, is there an alternative, and can it be avoided? Numbers were checked against the providers' own documentation and, where possible, live response headers on 2026-09-22.

## Literature discovery

| Service | Free? | Key? | Limits | Decision |
|---|---|---|---|---|
| **OpenAlex** | Yes. Usage-based pricing since Feb 2026, with a free daily allowance | Optional (free account). Keyless still works | Keyless $0.10/day; free key $1/day. Search $1/1k calls, list/filter $0.10/1k, single-entity lookups free (confirmed via `x-ratelimit-*-usd` headers: one filtered search cost $0.001) | **Primary source.** Uses the `title_and_abstract.search` filter, which gave far better precision than full-text `search` in testing. Key is optional (`CLAIMGRAPH_OPENALEX_API_KEY`) |
| **Semantic Scholar** | Yes | Optional (free, by request) | Keyless: shared pool, often throttled (live test returned 429); keyed: 1 req/s | **Secondary source.** 429s are retried and then skipped gracefully |
| **arXiv API** | Yes | No | ≤ 1 request / 3 s, one connection (Terms of Use) | **Used** for search and PDFs. The per-host limiter enforces 3 s |
| **Europe PMC** | Yes | No | Generous | **Used for biomedical questions** (added automatically), and its full-text JATS XML (OA subset) is the second acquisition route for any paper with a DOI |
| **Crossref** | Yes | No | Polite pool with mailto | Not needed: OpenAlex already ingests Crossref metadata and DOIs |
| **PubMed E-utilities** | Yes | Optional | 3 req/s keyless | Covered by Europe PMC, which indexes PubMed with better OA links |
| **Unpaywall** | Yes | No key; **email required** | 100k calls/day | **Used** for OA PDF lookup by DOI when `CLAIMGRAPH_CONTACT_EMAIL` is set |
| **Exa** | $10/month free credits after sign-up, then about $7 per 1k searches | **Yes**, account required | – | **Rejected.** Its publication search is good, but OpenAlex + S2 + arXiv cover discovery for free without an account. It can be added later as one more `LiteratureSource` |

Legal/ethical constraints implemented:
- only OA URLs surfaced by these APIs are fetched (no crawling)
- robots.txt is checked for non-arXiv and non-EPMC hosts
- per-host rate limiting, descriptive User-Agent, size-capped downloads, local cache so papers are fetched once

## PDF and document parsing

| Option | Notes | Decision |
|---|---|---|
| **PyMuPDF** | fast, accurate text and block geometry, pip-installable. AGPL (fine for an academic project; note it for commercial use) | **Used**, with a custom column-aware block ordering (the default sort interleaves two-column layouts) |
| pypdf | pure Python, weaker text ordering | fallback candidate only |
| GROBID | best structure (sections, references) but a Java service (Docker, ~1 GB RAM) | future improvement |
| Nougat / Marker | high quality, GPU-hungry | not practical on a CPU laptop |

## Local models

Target: a CPU-only machine (16 GB RAM, no GPU).

| Role | Model | Size | Why |
|---|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (ONNX) | ~130 MB | strong retrieval quality for its size; no PyTorch; MIT |
| Reranking | `Xenova/ms-marco-MiniLM-L-6-v2` via fastembed | ~90 MB | cross-encoder relevance for paper ranking |
| NLI | `cross-encoder/nli-deberta-v3-xsmall` (ONNX export in the repo), onnxruntime | ~280 MB | SNLI+MNLI; gives contradiction/entailment in about 10 ms per pair on CPU; Apache-2.0 |

- **sentence-transformers** was considered and rejected: it pulls in PyTorch (~2 GB) for the same models.
- **E5-small** was a viable alternative to bge-small.
- The larger `bge-reranker-base` / DeBERTa-base would improve quality at 3–4× the CPU cost.

## LLMs

| Option | Cost | Measured / notes | Decision |
|---|---|---|---|
| **Ollama** (local) | free | `qwen2.5:7b` on this CPU: **≈2.7 tokens/s**, ~80 s per structured call | Optional provider with a hard call budget. A 3B model is recommended on CPU |
| Hugging Face Inference Providers | free tier ≈ $0.10/month of router credit, then pay-as-you-go | OpenAI-compatible router | Optional `HuggingFaceProvider` |
| Any OpenAI-compatible API (OpenRouter, Groq, vLLM, LM Studio, llama.cpp) | varies; several have free tiers | one client | Optional `OpenAICompatibleProvider` |
| OpenAI / Anthropic direct | paid | – | Not integrated; reachable through `openai_compatible` if someone wants it |

**Consequence for the design:** at ~80 s per call, an LLM-first pipeline would take about 30 minutes for 8 papers. The core reasoning (extraction, NLI, stance, context comparison, rules) therefore runs on small local models, and the LLM is an optional *adjudicator and explainer* under a call budget. This also makes results more reproducible and inspectable.

## Backend and frontend

- **FastAPI + Pydantic v2:** typed schemas shared by the pipeline and the API, with automatic OpenAPI docs.
- **SQLite** over PostgreSQL: one JSON document per analysis and a single local user. PostgreSQL would add a service to run without adding a capability the app needs. SQLAlchemy keeps the switch to a URL change.
- **Background work:** an in-process single-worker thread pool. Celery or RQ would need a broker, which is unjustified for a local mini-project.
- **Frontend:** React + Vite + TypeScript. Streamlit was considered, but it can't deliver the interactive graph, side drawers or dense editorial layout.
- **Graph rendering:** Cytoscape.js for mature graph styling (dash patterns, arrows, preset and force layouts) at ~400 kB. React Flow and Sigma.js were alternatives.

## Sources

- OpenAlex pricing: https://help.openalex.org/access/pricing/ and https://help.openalex.org/access/example-costs/
- OpenAlex rate limits and authentication: https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication
- Semantic Scholar API: https://www.semanticscholar.org/product/api and https://api.semanticscholar.org/api-docs/
- arXiv API Terms of Use: https://info.arxiv.org/help/api/tou.html
- Unpaywall API: https://unpaywall.org/products/api
- Europe PMC REST API: https://europepmc.org/RestfulWebService
- Exa pricing: https://exa.ai/pricing
- Hugging Face Inference Providers: https://huggingface.co/docs/inference-providers/index
- fastembed supported models: https://qdrant.github.io/fastembed/
- NLI model card: https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall
