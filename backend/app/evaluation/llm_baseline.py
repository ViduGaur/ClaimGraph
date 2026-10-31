"""'Just ask an LLM' baselines: one call per pair returns the final label directly.

Two variants isolate what the hybrid's structure buys beyond the model itself:

* ``claims`` — the model sees only the two claim sentences;
* ``context`` — the model sees exactly what ClaimGraph's pair analyser sees (titles,
  abstracts, evidence passages), but decides end-to-end instead of filling a grounded
  context table for the rules.
"""
from __future__ import annotations

from app.reasoning.alignment import render_pair
from app.schemas.domain import Claim, Paper

DIRECT_LABELS = ["SUPPORTS", "EXTENDS", "QUALIFIES", "CONTRADICTS", "INSUFFICIENT", "UNRELATED"]

# the same worked examples the pair analyser gets (plus the labels it never has to name), so that the comparison
# is between architectures and not between prompt-engineering budgets
DIRECT_EXAMPLES: list[tuple[str, str, str]] = [
    ("Dropout improves the accuracy of CNNs on small image datasets.",
     "With billions of training images, dropout no longer improves accuracy.",
     "QUALIFIES (conflicting at face value, but the stated data scale differs)"),
    ("Fine-tuned BERT outperforms LSTMs on English sentiment classification.",
     "BERT also outperforms LSTMs on low-resource languages.", "EXTENDS"),
    ("BERT outperforms LSTMs on sentiment classification.", "BERT also outperforms LSTMs on sentiment classification in our replication.",
     "SUPPORTS"),
    ("In adults, drug D lowers blood pressure.", "In our trial with adults at the same dose, drug D did not lower blood pressure.",
     "CONTRADICTS"),
    ("Caffeine improves reaction time.", "Caffeine does not improve reaction time.",
     "INSUFFICIENT (they conflict, and neither states its conditions)"),
    ("Caffeine improves reaction time.", "Caffeine improves reaction time only in sleep-deprived participants.", "QUALIFIES"),
    ("Our pruning method retains 99% of the accuracy at 90% sparsity.", "Our quantisation method retains 99% of the accuracy at 8 bits.",
     "UNRELATED (different methods)"),
    ("Method X halves the training time.", "Method X improves accuracy by two points.", "UNRELATED (different outcomes)"),
]

DIRECT_SYSTEM = (
    "You are given two claims (A and B) from two different scientific papers. Classify how claim B relates to "
    "claim A. Use only the text you are given.\n"
    "- SUPPORTS: both reach the same conclusion about the same relationship, in the same or an unstated setting.\n"
    "- EXTENDS: both reach the same conclusion, and B establishes it in a different setting "
    "(another dataset, scale, task, population or condition).\n"
    "- CONTRADICTS: they reach conflicting conclusions about the same relationship under comparable conditions.\n"
    "- QUALIFIES: they conflict at face value, or one narrows the other, but the text states different conditions "
    "that could explain the difference.\n"
    "- INSUFFICIENT: they conflict at face value, but the text does not state enough about the conditions to tell "
    "whether they are comparable.\n"
    "- UNRELATED: they do not address the same relationship (sharing a topic is not enough).\n"
    "'The same relationship' means the same method, intervention or factor assessed on the same kind of outcome; a "
    "claim about one specific method and a claim about a different method are unrelated.\n\n"
    "Examples:\n"
    + "\n".join(f"- A: '{a}' B: '{b}' -> {label}" for a, b, label in DIRECT_EXAMPLES)
    + "\n\nGive the label and a one-sentence reason."
)

DIRECT_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": DIRECT_LABELS}, "reason": {"type": "string"}},
    "required": ["label", "reason"],
}


def direct_prompt(a: Claim, b: Claim, papers: dict[str, Paper], variant: str) -> str:
    if variant == "claims":
        return f"CLAIM A: {a.claim_text}\n\nCLAIM B: {b.claim_text}"
    return render_pair(a, b, papers)


def direct_label(out: dict | None) -> str | None:
    label = str((out or {}).get("label", "")).upper()
    return label if label in DIRECT_LABELS else None
