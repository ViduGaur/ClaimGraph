"""Local, CPU-friendly models: embeddings, reranking and NLI.

All models are small ONNX exports run with onnxruntime — no PyTorch, no GPU,
no API key. Each has a dependency-free fallback so the pipeline (and the test
suite) still runs when models cannot be downloaded:

* embeddings: ``BAAI/bge-small-en-v1.5`` (fastembed)   → hashing bag-of-ngrams
* reranking:  ``Xenova/ms-marco-MiniLM-L-6-v2``         → embedding cosine
* NLI:        ``cross-encoder/nli-deberta-v3-xsmall``   → lexical negation cues

Model files are cached in ``CLAIMGRAPH_MODEL_CACHE_DIR`` (project-local), never
in a shared/global Hugging Face cache, and no Hugging Face token is ever sent.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.config import Settings

log = logging.getLogger(__name__)
_LOCK = threading.Lock()


def _isolate_hf_cache(cache_dir: Path) -> None:
    """Point the HF hub at our own cache for this process only."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# --------------------------------------------------------------------------- embeddings
class Embedder(ABC):
    name: str

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised vectors, shape (n, d)."""


class HashingEmbedder(Embedder):
    """Deterministic bag of word uni/bi-grams hashed into a fixed space."""

    name = "hashing"
    _STOP = frozenset(
        "a an the of on in for to and or with by from as at is are was were be been this that these those "
        "we our it its their they which when while than can may also such".split()
    )

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        words = [w for w in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()) if w not in self._STOP]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in self._tokens(text):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class FastEmbedEmbedder(Embedder):
    name = "fastembed"

    def __init__(self, model_name: str, cache_dir: Path):
        _isolate_hf_cache(cache_dir)
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vecs = np.array(list(self._model.embed(texts, batch_size=32)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)


# --------------------------------------------------------------------------- reranking
class Reranker(ABC):
    name: str

    @abstractmethod
    def score(self, query: str, docs: list[str]) -> list[float]:
        """Relevance in [0, 1] for each doc."""


class EmbeddingReranker(Reranker):
    name = "embedding-cosine"

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        vecs = self.embedder.embed([query, *docs])
        sims = vecs[1:] @ vecs[0]
        return [float((s + 1) / 2) for s in sims]


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"

    def __init__(self, model_name: str, cache_dir: Path):
        _isolate_hf_cache(cache_dir)
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir))

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        logits = np.array(list(self._model.rerank(query, docs)), dtype=np.float32)
        return [float(x) for x in 1 / (1 + np.exp(-logits))]


# --------------------------------------------------------------------------- NLI
class NLIModel(ABC):
    name: str

    @abstractmethod
    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """For (premise, hypothesis) pairs return {entailment, contradiction, neutral} probabilities."""


class LexicalNLI(NLIModel):
    """Crude fallback: overlap ⇒ entailment, overlap + negation mismatch ⇒ contradiction."""

    name = "lexical"
    _NEG = re.compile(
        r"\b(not|no|never|cannot|fail(s|ed)?|without|worse|underperform\w*|inferior|lack\w*|unable|"
        r"not necessary|insufficient|degrad\w*|challenge|dispel|myth)\b",
        re.I,
    )

    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        emb = HashingEmbedder()
        out = []
        for p, h in pairs:
            v = emb.embed([p, h])
            sim = float(max(0.0, v[0] @ v[1]))
            neg_mismatch = bool(self._NEG.search(p)) != bool(self._NEG.search(h))
            if neg_mismatch:
                c, e = 0.25 + 0.5 * sim, 0.1 * sim
            else:
                c, e = 0.05, 0.2 + 0.6 * sim
            n = max(0.0, 1.0 - c - e)
            total = c + e + n
            out.append({"entailment": e / total, "contradiction": c / total, "neutral": n / total})
        return out


class OnnxNLI(NLIModel):
    """DeBERTa-v3 cross-encoder trained on SNLI + MultiNLI, run with onnxruntime."""

    name = "onnx"

    def __init__(self, repo_id: str, cache_dir: Path):
        _isolate_hf_cache(cache_dir)
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        kw = {"cache_dir": str(cache_dir / "hub"), "token": False}
        model_path = hf_hub_download(repo_id, "onnx/model.onnx", **kw)
        tok_path = hf_hub_download(repo_id, "tokenizer.json", **kw)
        cfg = json.loads(Path(hf_hub_download(repo_id, "config.json", **kw)).read_text())
        self.labels = {int(k): v.lower() for k, v in cfg["id2label"].items()}
        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        for start in range(0, len(pairs), 16):
            batch = pairs[start : start + 16]
            enc = self.tokenizer.encode_batch(batch)
            feed = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            }
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            logits = self.session.run(None, {k: v for k, v in feed.items() if k in self.input_names})[0]
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            for row in probs:
                results.append({self.labels[i]: float(p) for i, p in enumerate(row)})
        return results


# --------------------------------------------------------------------------- registry
class ModelBundle:
    """The three local models used by one analysis, with fallback bookkeeping."""

    def __init__(self, embedder: Embedder, reranker: Reranker, nli: NLIModel, notes: list[str]):
        self.embedder = embedder
        self.reranker = reranker
        self.nli = nli
        self.notes = notes


@lru_cache(maxsize=4)
def _load_bundle(embedding_backend: str, embedding_model: str, reranker_model: str, nli_backend: str,
                 nli_model: str, cache_dir: str) -> ModelBundle:
    notes: list[str] = []
    cache = Path(cache_dir)
    embedder: Embedder
    reranker: Reranker
    nli: NLIModel
    if embedding_backend == "fastembed":
        try:
            embedder = FastEmbedEmbedder(embedding_model, cache)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"embedding model {embedding_model} unavailable ({exc.__class__.__name__}); using hashing embeddings")
            embedder = HashingEmbedder()
    else:
        embedder = HashingEmbedder()
    reranker = EmbeddingReranker(embedder)
    if embedding_backend == "fastembed" and reranker_model:
        try:
            reranker = CrossEncoderReranker(reranker_model, cache)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"reranker {reranker_model} unavailable ({exc.__class__.__name__}); using embedding cosine")
    if nli_backend == "onnx":
        try:
            nli = OnnxNLI(nli_model, cache)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"NLI model {nli_model} unavailable ({exc.__class__.__name__}); using lexical NLI heuristics")
            nli = LexicalNLI()
    else:
        nli = LexicalNLI()
    return ModelBundle(embedder, reranker, nli, notes)


def get_models(settings: Settings) -> ModelBundle:
    with _LOCK:
        return _load_bundle(
            settings.embedding_backend,
            settings.embedding_model,
            settings.reranker_model,
            settings.nli_backend,
            settings.nli_model,
            str(settings.model_cache_dir),
        )
