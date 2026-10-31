"""Optional: exercises the real ONNX models if they are already cached locally."""
import pytest

from app.core.config import Settings

settings = Settings()
cached = (settings.model_cache_dir / "hub").exists() or any(settings.model_cache_dir.glob("*bge*"))
pytestmark = [pytest.mark.models, pytest.mark.skipif(not cached, reason="local models not downloaded")]


def test_onnx_nli_detects_contradiction():
    from app.providers.models import get_models

    m = get_models(settings)
    assert m.nli.name == "onnx"
    out = m.nli.predict([
        ("Vision transformers need large-scale pre-training to beat CNNs.",
         "Vision transformers beat CNNs without any pre-training on small datasets."),
        ("ViT models need large pre-training datasets.", "ViTs require big pre-training data."),
    ])
    assert out[0]["contradiction"] > 0.5
    assert out[1]["entailment"] > out[1]["contradiction"]
