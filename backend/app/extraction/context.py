"""Rule-based extraction of experimental context from claim/evidence text.

This is deliberately transparent: every extracted value is a literal span (or a
canonicalised alias of one) found in the source text, so it can be shown to the
user as evidence. Empty fields mean "not stated", which conflict analysis
treats as *unknown*, never as *different*.
"""
from __future__ import annotations

import re

from app.schemas.domain import ClaimContext

# canonical name -> alias regex. Order matters: more specific first.
ENTITY_ALIASES: list[tuple[str, str]] = [
    ("ViT", r"vision transformers?|visual transformers?|\bViTs?\b|\bVTs\b|plain ViT|vanilla ViT"),
    ("DeiT", r"\bDeiT\b"),
    ("Swin Transformer", r"swin transformers?|\bSwin\b"),
    ("CCT", r"compact (convolutional )?transformers?|\bCCT\b"),
    ("MLP-Mixer", r"MLP-Mixers?|\bMixer-B/16\b"),
    ("ConvNeXt", r"ConvNeXts?"),
    ("NFNet", r"NFNets?"),
    ("ResNet", r"ResNet-?\d*s?\b"),
    ("CNN", r"convolutional (neural )?networks?|\bconvnets?\b|\bCNNs?\b|\bConvNets?\b"),
    ("Transformer", r"\btransformers?\b"),
    ("BERT", r"\bBERT\b"),
    ("GPT", r"\bGPT-?\d?(\.\d)?\b"),
    ("LSTM", r"\bLSTMs?\b"),
    ("LLM", r"large language models?|\bLLMs?\b"),
    ("random forest", r"random forests?"),
    ("XGBoost", r"\bXGBoost\b"),
    ("logistic regression", r"logistic regression"),
    ("SVM", r"\bSVMs?\b|support vector machines?"),
]
_ENTITY_RES = [(name, re.compile(pat, re.I)) for name, pat in ENTITY_ALIASES]
# vision/general architecture names that look like CamelCase model identifiers
_MODEL_ID = re.compile(r"\b(?:[A-Z][a-z]+[A-Z][A-Za-z]*|[A-Z][A-Za-z]*-[A-Z]?\d+[A-Za-z/\d]*)\b")
_NOT_MODELS = {"ImageNet", "JavaScript", "GitHub", "TensorFlow", "PyTorch", "OpenAlex", "COVID-19", "SARS-CoV-2",
               "CIFAR-10", "CIFAR-100", "JFT-300M", "JFT-4B", "JFT-3B", "ImageNet-21k", "ImageNet-1k",
               "TPUv3-8", "TPU-v4", "AugReg", "TPUv3", "NAS-based", "Tiny-ImageNet", "Flowers-102", "ADE20K", "ResNet50"}

# method acronyms used across fields (CoT, IRCoT, CLP, SVM, RNN, GAN, PPO, CBT, ...)
_ACRONYM = re.compile(r"\b(?:[A-Z]{2,}[a-z]?[A-Z]*\+?|[A-Z][a-z]?[A-Z]{2,}\+?|[A-Z][a-z][A-Z][a-z]?[A-Z]?)(?![\w-])")
_NOT_ACRONYMS = {"QA", "NLP", "CV", "AI", "ML", "OOD", "SOTA", "GPU", "GPUs", "TPU", "TPUs", "CPU", "USA", "UK", "EU", "US",
                 "II", "III", "IV", "PDF", "API", "URL", "ID", "DNA", "RNA", "HIV", "COVID", "WHO", "FDA", "RCT", "RCTs",
                 "BMI", "CI", "OR", "HR", "SD", "SE", "IQR", "NA", "PhD", "MSc", "BSc", "Table", "Fig", "LaTeX", "OA",
                 "IEEE", "ACM", "ACL", "EMNLP", "NAACL", "ICLR", "ICML", "CVPR", "ICCV", "ECCV", "AAAI", "IJCAI", "NeurIPS"}


def _metric_like(tok: str) -> bool:
    return any(rx.fullmatch(tok) for _, rx in _METRIC_RES)


DATASETS = [
    ("ImageNet-21k", r"ImageNet-?21k|ImageNet-?22k"),
    ("ImageNet-1k", r"ImageNet-?1k|(?<![-\w])ImageNet\b(?!-)"),
    ("Tiny-ImageNet", r"Tiny-?ImageNet"),
    ("JFT-300M", r"JFT-?300M"),
    ("JFT-3B", r"JFT-?3B"),
    ("JFT-4B", r"JFT-?4B"),
    ("CIFAR-10", r"CIFAR-?10\b"),
    ("CIFAR-100", r"CIFAR-?100\b"),
    ("MNIST", r"(?<!-)\bMNIST\b"),
    ("Fashion-MNIST", r"Fashion-?MNIST"),
    ("SVHN", r"\bSVHN\b"),
    ("COCO", r"\b(MS-?)?COCO\b"),
    ("ADE20K", r"ADE-?20K"),
    ("VTAB", r"\bVTAB\b"),
    ("Flowers-102", r"Flowers-?102"),
    ("Oxford Pets", r"Oxford(-IIIT)? Pets?"),
    ("iNaturalist", r"iNaturalist"),
    ("GLUE", r"\bGLUE\b"),
    ("SuperGLUE", r"SuperGLUE"),
    ("SQuAD", r"SQuAD"),
    ("MMLU", r"\bMMLU\b"),
    ("GSM8K", r"GSM-?8K"),
    ("HumanEval", r"HumanEval"),
    ("WikiText-103", r"WikiText-?103"),
    ("LibriSpeech", r"LibriSpeech"),
    ("MIMIC", r"MIMIC(-I{2,3}|-IV)?"),
    ("UK Biobank", r"UK Biobank"),
]
_DATASET_RES = [(name, re.compile(pat, re.I)) for name, pat in DATASETS]
_GENERIC_DATASET = re.compile(r"\b(?:the\s+)?([A-Z][A-Za-z0-9\-]{2,25})\s+(?:dataset|benchmark|corpus|cohort)\b")

METRICS = [
    ("top-1 accuracy", r"top-?1( accuracy)?"),
    ("top-5 accuracy", r"top-?5( accuracy)?"),
    ("accuracy", r"\baccurac(y|ies)\b"),
    ("F1", r"\bF1\b|F-?score"),
    ("precision", r"\bprecision\b"),
    ("recall", r"\brecall\b"),
    ("AUC", r"\bAUC\b|\bAUROC\b|area under the (ROC )?curve"),
    ("mAP", r"\bmAP\b|mean average precision"),
    ("BLEU", r"\bBLEU\b"),
    ("ROUGE", r"\bROUGE(-[L12])?\b"),
    ("perplexity", r"perplexity"),
    ("loss", r"\b(?:held[- ]out|validation|test|training|train|cross[- ]entropy)\s+loss\b"),
    ("RMSE", r"\bRMSE\b|root mean squared? error"),
    ("MAE", r"\bMAE\b|mean absolute error"),
    ("robustness", r"\brobust(ness)?\b"),
    ("throughput", r"\bthroughput\b"),
    ("compute cost", r"\bcompute( budget)?s?\b|\bFLOPs\b|TPU[- ]?(v\d)?[- ]?(core )?(hours|days)|computational resources"),
    ("parameter count", r"\bparameters?\b"),
    ("mortality", r"\bmortality\b"),
    ("odds ratio", r"odds ratio"),
    ("hazard ratio", r"hazard ratio"),
    ("effect size", r"effect size"),
]
_METRIC_RES = [(name, re.compile(pat, re.I)) for name, pat in METRICS]

CONDITIONS = [
    ("large-scale pre-training", r"large[- ]scale pre-?train\w*|pre-?trained on (large|hundreds of millions|web)|large amounts of data|web-?scale|pre-?training using a large"),
    ("pre-training", r"\bpre-?train(ed|ing)?\b"),
    ("trained from scratch", r"from scratch"),
    ("no external data", r"no external data|(imagenet|trained on \w+) only|without (large-scale )?pre-?training"),
    ("fine-tuning", r"fine-?tun\w*"),
    ("transfer learning", r"transfer(red|ring)? (learning|to)"),
    ("small datasets", r"small(-size|er)?[- ](training )?(data ?sets?|data|sets of data|training[- ]sets?)|data[- ]scarce|scarce (training )?data|small training-set|limited (training )?data"),
    ("mid-sized datasets", r"mid-sized|moderately sized|medium-sized"),
    ("zero-shot", r"zero-?shot"),
    ("few-shot", r"few-?shot"),
    ("self-supervised", r"self-?supervised"),
    ("supervised", r"\bsupervised\b"),
]
_CONDITION_RES = [(name, re.compile(pat, re.I)) for name, pat in CONDITIONS]

TRAINING = [
    ("data augmentation", r"(data )?augmentations?|AugReg"),
    ("regularization", r"regulari[sz]ation"),
    ("knowledge distillation", r"distillation"),
    ("sharpness-aware optimization", r"sharpness-?aware|\bSAM\b"),
    ("convolutional tokenization", r"convolutional tokeni[sz]ation"),
    ("locality inductive bias", r"locality|inductive bias"),
    ("increased compute", r"increased compute|compute budgets?"),
    ("longer training", r"\d+ epochs"),
]
_TRAINING_RES = [(name, re.compile(pat, re.I)) for name, pat in TRAINING]

TASKS = [
    ("image classification", r"image (recognition|classification)|classification"),
    ("object detection", r"object detection|\bdetection\b"),
    ("semantic segmentation", r"segmentation"),
    ("question answering", r"question answering"),
    ("machine translation", r"machine translation"),
    ("summarization", r"summari[sz]ation"),
    ("language modelling", r"language model(l)?ing"),
    ("diagnosis", r"diagnos(is|tic)"),
]
_TASK_RES = [(name, re.compile(pat, re.I)) for name, pat in TASKS]

POPULATION = re.compile(r"\b(patients|participants|adults|children|adolescents|women|men|infants|elderly|older adults|"
                        r"mice|rats|students|volunteers|subjects)\b", re.I)
SAMPLE_SIZE = re.compile(r"\b(?:n\s*=\s*[\d,]+|[\d,]{2,}\s+(?:patients|participants|subjects|images|samples|examples|adults|children))\b", re.I)
METHODOLOGY = re.compile(r"(randomi[sz]ed controlled trial|meta-analysis|systematic (empirical )?(study|review)|cohort study|"
                         r"cross-sectional|case-control|ablation|empirical(ly)? analy[sz]\w*|theoretical analysis|simulation study)", re.I)
# surface forms → one label, so "empirically analyse" vs "systematic empirical study" is not reported as a difference
_METHODOLOGY_CANON = [
    (re.compile(r"randomi[sz]ed controlled trial", re.I), "randomised controlled trial"),
    (re.compile(r"systematic review", re.I), "systematic review"),
    (re.compile(r"empirical|systematic (empirical )?study", re.I), "empirical study"),
]


def canonical_methodology(text: str) -> str:
    for pattern, label in _METHODOLOGY_CANON:
        if pattern.search(text):
            return label
    return text.lower()


TIME_PERIOD = re.compile(r"\b(?:between|from)\s+(19|20)\d{2}\s+(?:and|to)\s+(19|20)\d{2}\b")
BASELINE = re.compile(r"(?:than|compared (?:to|with)|over|against|relative to)\s+((?:the\s+)?[A-Za-z][\w\-/ ]{2,40}?)(?=[,.;)]|\s+(?:on|in|with|at|while|when|by|and)\b|$)", re.I)

DATA_SCALE = {
    "small": {"CIFAR-10", "CIFAR-100", "Tiny-ImageNet", "Flowers-102", "Oxford Pets", "MNIST", "Fashion-MNIST", "SVHN"},
    "medium": {"ImageNet-1k"},
    "large": {"ImageNet-21k"},
    "web-scale": {"JFT-300M", "JFT-3B", "JFT-4B"},
}
_SCALE_ORDER = ["small", "medium", "large", "web-scale"]


def canonical_entities(text: str) -> list[str]:
    """Model/architecture entities mentioned in text, canonicalised, in order of appearance."""
    hits: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    for name, rx in _ENTITY_RES:
        for m in rx.finditer(text):
            if any(s <= m.start() < e for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            hits.append((m.start(), name))
    for m in _ACRONYM.finditer(text):
        tok = m.group(0)
        if tok in _NOT_ACRONYMS or any(s <= m.start() < e for s, e in taken) or _dataset_name(tok) or _metric_like(tok):
            continue
        taken.append((m.start(), m.end()))
        hits.append((m.start(), tok))
    for m in _MODEL_ID.finditer(text):
        tok = m.group(0)
        if tok in _NOT_MODELS or any(s <= m.start() < e for s, e in taken) or _dataset_name(tok):
            continue
        hits.append((m.start(), tok))
    seen: list[str] = []
    for _, name in sorted(hits):
        if name not in seen:
            seen.append(name)
    return seen


def _dataset_name(token: str) -> str | None:
    for name, rx in _DATASET_RES:
        if rx.fullmatch(token):
            return name
    return None


def _collect(res: list[tuple[str, re.Pattern]], text: str) -> list[str]:
    return [name for name, rx in res if rx.search(text)]


def extract_datasets(text: str) -> list[str]:
    found = _collect(_DATASET_RES, text)
    for m in _GENERIC_DATASET.finditer(text):
        cand = m.group(1)
        if cand not in found and cand.lower() not in {"this", "that", "our", "the", "large", "small", "public", "same", "each"}:
            found.append(cand)
    # "ImageNet-21k" text also matches the ImageNet-1k fallback pattern only when bare
    if "ImageNet-21k" in found and "ImageNet-1k" in found and not re.search(r"ImageNet-?1k|\bImageNet\b(?![-‑]?2[12]k)", text, re.I):
        found.remove("ImageNet-1k")
    return found


def infer_data_scale(datasets: list[str], conditions: list[str], text: str) -> str | None:
    scales = [s for s in _SCALE_ORDER for d in datasets if d in DATA_SCALE[s]]
    if "large-scale pre-training" in conditions or re.search(r"web-?scale|hundreds of millions", text, re.I):
        scales.append("web-scale" if re.search(r"web|JFT|hundreds of millions", text, re.I) else "large")
    if "small datasets" in conditions:
        scales.append("small")
    if "mid-sized datasets" in conditions:
        scales.append("medium")
    if not scales:
        return None
    # the largest *training* scale mentioned dominates (pre-train on big, transfer to small ⇒ big)
    return max(scales, key=_SCALE_ORDER.index)


def extract_context(text: str) -> ClaimContext:
    conditions = _collect(_CONDITION_RES, text)
    if "large-scale pre-training" in conditions and "pre-training" in conditions:
        conditions.remove("pre-training")
    if re.search(r"without (large-scale )?pre-?train", text, re.I):
        conditions = [c for c in conditions if c not in ("pre-training", "large-scale pre-training")]
    datasets = extract_datasets(text)
    models = canonical_entities(text)
    baselines = []
    for m in BASELINE.finditer(text):
        ents = canonical_entities(m.group(1))
        baselines.extend(e for e in ents if e not in baselines)
    sample = SAMPLE_SIZE.search(text)
    method = METHODOLOGY.search(text)
    period = TIME_PERIOD.search(text)
    return ClaimContext(
        datasets=datasets,
        models=models,
        metrics=_collect(_METRIC_RES, text),
        population=sorted({m.group(1).lower() for m in POPULATION.finditer(text)}),
        sample_size=sample.group(0) if sample else None,
        conditions=conditions,
        training=_collect(_TRAINING_RES, text),
        baselines=baselines,
        task=_collect(_TASK_RES, text),
        data_scale=infer_data_scale(datasets, conditions, text),
        time_period=period.group(0) if period else None,
        methodology=canonical_methodology(method.group(0)) if method else None,
    )


def merge_context(primary: ClaimContext, fallback: ClaimContext) -> tuple[ClaimContext, list[str]]:
    """Fill fields the claim sentence leaves empty from surrounding/paper-level context."""
    data = primary.model_dump()
    inherited = []
    for key, value in fallback.model_dump().items():
        if value and not data.get(key):
            data[key] = value
            inherited.append(key)
    return ClaimContext(**data), inherited
