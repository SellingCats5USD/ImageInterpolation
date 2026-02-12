import torch

import src.model as model_mod
from src.model import _load_pipeline_with_fallback, _load_sdxl_pipeline


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


def test_load_sdxl_pipeline_falls_back_to_single_file_for_juggernaut_lightning(monkeypatch) -> None:
    calls: list[tuple[str, str, object]] = []

    def _fake_load_pipeline_with_fallback(*args, **kwargs):
        raise OSError("Could not find the necessary safetensors weights")

    def _fake_hf_hub_download(repo_id: str, filename: str) -> str:
        calls.append(("download", repo_id, filename))
        return "/tmp/juggernaut.safetensors"

    class _FakeSDXLPipeline:
        @classmethod
        def from_single_file(cls, path: str, torch_dtype):
            calls.append(("single_file", path, torch_dtype))
            return object()

    monkeypatch.setattr(model_mod, "_load_pipeline_with_fallback", _fake_load_pipeline_with_fallback)
    monkeypatch.setattr(model_mod, "hf_hub_download", _fake_hf_hub_download)
    monkeypatch.setattr(model_mod, "StableDiffusionXLPipeline", _FakeSDXLPipeline)

    result = _load_sdxl_pipeline(model_mod.JUGGERNAUT_LIGHTNING_REPO, torch.float16)
    assert result is not None
    assert calls[0] == ("download", model_mod.JUGGERNAUT_LIGHTNING_REPO, model_mod.JUGGERNAUT_LIGHTNING_SINGLE_FILE)
    assert calls[1] == ("single_file", "/tmp/juggernaut.safetensors", torch.float16)
