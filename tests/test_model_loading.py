import torch

from src.model import _load_pipeline_with_fallback


class _FakePipeline:
    calls = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append((model_id, kwargs))
        if len(cls.calls) == 1:
            raise OSError("Could not find the necessary safetensors weights")
        return object()


def test_load_pipeline_retries_fp16_variant_on_safetensor_error() -> None:
    _FakePipeline.calls.clear()
    result = _load_pipeline_with_fallback(_FakePipeline, "foo/bar", torch.float16)
    assert result is not None
    assert len(_FakePipeline.calls) == 2
    assert _FakePipeline.calls[0][1].get("variant") is None
    assert _FakePipeline.calls[1][1]["variant"] == "fp16"
