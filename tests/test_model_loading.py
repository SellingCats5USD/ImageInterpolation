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


class _FakePipelineForceThreeAttempts:
    calls = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append((model_id, kwargs))
        if len(cls.calls) <= 2:
            raise OSError("Could not find the necessary safetensors weights")
        return object()


def test_load_pipeline_retries_without_safetensors_when_forced() -> None:
    _FakePipeline.calls.clear()
    result = _load_pipeline_with_fallback(_FakePipeline, "foo/bar", torch.float32, use_safetensors=True)
    assert result is not None
    assert len(_FakePipeline.calls) == 2
    assert _FakePipeline.calls[0][1]["use_safetensors"] is True
    assert _FakePipeline.calls[1][1]["use_safetensors"] is False


def test_load_pipeline_retries_fp16_variant_on_safetensor_error() -> None:
    _FakePipelineForceThreeAttempts.calls.clear()
    result = _load_pipeline_with_fallback(
        _FakePipelineForceThreeAttempts,
        "foo/bar",
        torch.float16,
        use_safetensors=True,
    )
    assert result is not None
    assert len(_FakePipelineForceThreeAttempts.calls) == 3

    first_kwargs = _FakePipelineForceThreeAttempts.calls[0][1]
    second_kwargs = _FakePipelineForceThreeAttempts.calls[1][1]
    third_kwargs = _FakePipelineForceThreeAttempts.calls[2][1]

    assert first_kwargs["use_safetensors"] is True
    assert first_kwargs.get("variant") is None
    assert second_kwargs["use_safetensors"] is False
    assert second_kwargs.get("variant") is None
    assert third_kwargs["use_safetensors"] is True
    assert third_kwargs["variant"] == "fp16"
