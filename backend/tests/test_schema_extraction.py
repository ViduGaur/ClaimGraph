"""Claim schema validation and claim/context extraction."""
import numpy as np
import pytest
from pydantic import ValidationError

from app.extraction.acquisition import PaperText, reconcile_abstract
from app.extraction.claims import (
    ClaimExtractor,
    classify_type,
    detect_polarity,
    is_clean_sentence,
    normalize_claim_text,
    subject_and_object,
)
from app.extraction.context import canonical_entities, canonical_methodology, extract_context
from app.extraction.text import Section, classify_heading, sections_from_text, split_sentences
from app.schemas.domain import Claim, ClaimType, Paper, Polarity
from tests.conftest import make_claim, make_paper


# ------------------------------------------------------------------ schema
def test_claim_requires_evidence():
    with pytest.raises(ValidationError):
        Claim(claim_id="c", paper_id="p", claim_text="ViTs outperform CNNs on ImageNet.", normalized_text="x",
              claim_type=ClaimType.FINDING, evidence=[], section="abstract", confidence=0.5, source_citation="A (2020)")


def test_claim_rejects_trivial_text_and_bad_confidence():
    with pytest.raises(ValidationError):
        make_claim("c", "p", "Too short")
    with pytest.raises(ValidationError):
        make_claim("c", "p", "A perfectly fine claim sentence here.", confidence=1.5)


def test_paper_doi_normalization_and_citation():
    p = Paper(paper_id="x", title="T", doi="https://doi.org/10.1000/ABC", authors=["Ada Lovelace", "B"], year=1843)
    assert p.doi == "10.1000/abc"
    assert p.citation == "Lovelace et al. (1843)"


# ------------------------------------------------------------------ text
def test_sentence_splitter_handles_abbreviations_and_decimals():
    text = "ViT reaches 88.5% top-1 accuracy, e.g. on ImageNet. Dosovitskiy et al. report this. It is fast."
    assert split_sentences(text) == [
        "ViT reaches 88.5% top-1 accuracy, e.g. on ImageNet.",
        "Dosovitskiy et al. report this.",
        "It is fast.",
    ]


def test_heading_detection_and_reference_cutoff():
    assert classify_heading("4 Experiments") == "experiments"
    assert classify_heading("5.1 Main Results") == "results"
    assert classify_heading("Model Output") is None  # figure labels are not the Methods heading
    raw = "Abstract\nWe study X.\n1 Introduction\nX matters.\n6 Conclusion\nX works.\nReferences\n[1] Someone. 2020."
    secs = sections_from_text(raw)
    assert [s.name for s in secs] == ["abstract", "introduction", "conclusion"]
    assert "Someone" not in " ".join(s.text for s in secs)


def test_reconcile_abstract_prefers_metadata():
    secs = [Section("abstract", "x" * 5000), Section("results", "r")]
    out = reconcile_abstract(secs, "Clean abstract text.")
    assert out[0] == Section("abstract", "Clean abstract text.")
    assert [s.name for s in out] == ["abstract", "introduction", "results"]


def test_clean_sentence_filter():
    assert is_clean_sentence("Chain-of-thought prompting improves arithmetic reasoning in large models.")
    assert not is_clean_sentence("12.3 45.1 67.8 90.2 11.4 Table 2 results 5 6 7")
    # words glued across merged PDF columns (lower→Upper join) are rejected
    assert not is_clean_sentence("PS prompting improves reasoningPerformance and then derives an answer by executing it.")


# ------------------------------------------------------------------ context
def test_context_extraction():
    ctx = extract_context("When pre-trained on JFT-300M and fine-tuned on CIFAR-10, ViT-B/16 beats ResNet-50 in top-1 accuracy "
                          "on image classification with 1,000 images.")
    assert "JFT-300M" in ctx.datasets and "CIFAR-10" in ctx.datasets
    assert ctx.data_scale == "web-scale"
    assert "top-1 accuracy" in ctx.metrics
    assert "fine-tuning" in ctx.conditions
    assert "image classification" in ctx.task
    assert ctx.sample_size == "1,000 images"


def test_tiny_imagenet_is_not_imagenet():
    assert extract_context("Results on Tiny-ImageNet").datasets == ["Tiny-ImageNet"]


def test_entities_are_canonicalised():
    assert canonical_entities("Vision Transformers and ConvNets and CNNs") == ["ViT", "CNN"]
    assert "CoT" in canonical_entities("Self-consistency improves CoT prompting")


def test_without_pretraining_is_not_pretraining():
    ctx = extract_context("ViTs outperform ResNets when trained from scratch on ImageNet without large-scale pre-training.")
    assert "large-scale pre-training" not in ctx.conditions
    assert "trained from scratch" in ctx.conditions


# ------------------------------------------------------------------ claim structure
@pytest.mark.parametrize("sentence,section,expected", [
    ("We show that ViTs outperform ResNets on ImageNet.", "abstract", ClaimType.COMPARISON),
    ("Our model achieves 83.1% top-1 accuracy on ImageNet.", "results", ClaimType.FINDING),
    ("We propose a new tokenization scheme for images.", "abstract", ClaimType.METHOD),
    ("Future work should study larger models.", "conclusion", ClaimType.FUTURE_WORK),
    ("It is commonly accepted that ViTs require heavy regularization.", "abstract", ClaimType.BACKGROUND),
    ("However, our approach fails to generalize to video.", "discussion", ClaimType.LIMITATION),
    ("Code is available at https://github.com/x/y.", "abstract", None),
])
def test_claim_type_classification(sentence, section, expected):
    assert classify_type(sentence, section) == expected


def test_polarity_and_negation():
    assert detect_polarity("ViTs outperform CNNs") == Polarity.POSITIVE
    assert detect_polarity("ViTs are more data-hungry than CNNs") == Polarity.NEGATIVE
    assert detect_polarity("NFNets match the performance of ViTs") == Polarity.NEUTRAL
    assert detect_polarity("ViTs do not outperform CNNs") == Polarity.NEGATIVE


def test_subject_and_object():
    assert subject_and_object("The resultant ViTs outperform ResNets of similar size.") == ("ViT", "ResNet")
    assert subject_and_object("Transformers can outperform state-of-the-art CNNs on small datasets.") == ("Transformer", "CNN")
    subj, _ = subject_and_object("Our best model reaches 98% accuracy on CIFAR-10.", "Escaping the Big Data Paradigm with Compact Transformers")
    assert subj == "CCT"


def test_normalization_strips_frames_and_aliases():
    assert normalize_claim_text("We show that vision transformers beat ConvNets.") == "ViT beat CNN"


# ------------------------------------------------------------------ extractor
def test_extractor_produces_grounded_claims(models, no_llm):
    abstract = ("Vision transformers are widely used. We show that vision transformers outperform CNNs on CIFAR-10 when trained "
                "from scratch with strong augmentation. Our model reaches 98% top-1 accuracy on CIFAR-10. "
                "Code is available at https://github.com/x. Future work should explore video recognition tasks.")
    paper = make_paper("p1").model_copy(update={"title": "Compact vision transformers", "abstract": abstract, "text_source": "abstract"})
    q = models.embedder.embed(["Do vision transformers beat CNNs on small datasets?"])[0]
    claims, _ = ClaimExtractor(models, no_llm).extract("Do vision transformers beat CNNs on small datasets?", q,
                                                       PaperText(paper, [Section("abstract", abstract)]))
    assert claims, "expected at least one claim"
    for c in claims:
        assert c.claim_text in c.evidence[0].text  # verbatim grounding
        assert c.paper_id == "p1"
        assert "github" not in c.claim_text
        assert any("abstract only" in n for n in c.uncertainty_notes)
    comp = next(c for c in claims if "outperform" in c.claim_text)
    assert comp.claim_type == ClaimType.COMPARISON
    assert comp.subject == "ViT" and "CNN" in comp.context.baselines
    assert "CIFAR-10" in comp.context.datasets


def test_extractor_dedupes_repeated_claims(models, no_llm):
    text = "We show that ViTs outperform CNNs on CIFAR-10. We show that ViTs outperform CNNs on CIFAR-10."
    paper = make_paper("p2").model_copy(update={"abstract": text})
    q = models.embedder.embed(["vit cnn"])[0]
    claims, _ = ClaimExtractor(models, no_llm).extract("vit cnn", q, PaperText(paper, [Section("abstract", text)]))
    assert len(claims) == 1


def test_extractor_empty_text(models, no_llm):
    paper = make_paper("p3")
    claims, warnings = ClaimExtractor(models, no_llm).extract("q", np.zeros(1024, dtype=np.float32), PaperText(paper, []))
    assert claims == [] and warnings


def test_jats_parsing_keeps_text_after_citations():
    from app.extraction.text import jats_to_sections

    xml = (b"<article><front><article-meta><abstract><p>Fasting helps.</p></abstract></article-meta></front><body>"
           b"<sec><title>1. Introduction</title><p>Obesity is common [<xref>1</xref>]. Due to cost, many trials exist "
           + b"and more words follow here. " * 20 +
           b"</p></sec><sec><title>Results</title><p>Insulin sensitivity improved by 12%.<fig><caption>x</caption></fig> After</p></sec>"
           b"<back><ref-list><ref>Someone 2020</ref></ref-list></back></body></article>")
    secs = jats_to_sections(xml)
    assert [s.name for s in secs] == ["abstract", "introduction", "results"]
    assert "Obesity is common. Due to cost" in secs[1].text
    assert secs[2].text.endswith("improved by 12%. After")
    assert "Someone" not in " ".join(s.text for s in secs)


def test_running_headers_removed():
    from app.extraction.text import _strip_running_headers

    bodies = ["Fasting improved insulin sensitivity.", "Weight loss was similar.", "Adherence was lower.",
              "Lipids did not change.", "Further trials are needed."]
    pages = [f"Int. J. of Research 2026\n{b}\n{i}" for i, b in enumerate(bodies, 1)]
    out = _strip_running_headers(pages)
    assert all("Int. J." not in p for p in out)
    assert out[2] == "Adherence was lower."


def test_methodology_surface_forms_share_one_label():
    assert canonical_methodology("empirically analyse") == canonical_methodology("systematic empirical study") == "empirical study"
    assert canonical_methodology("Randomized controlled trial") == "randomised controlled trial"
    assert canonical_methodology("meta-analysis") == "meta-analysis"
    assert extract_context("We empirically analyse ViT robustness.").methodology == "empirical study"
