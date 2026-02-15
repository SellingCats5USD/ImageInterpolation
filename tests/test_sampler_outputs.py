import torch

import src.sampler as sampler
from src.sampler import SampleConfig, sample_visual_anagram


class _DummyImageProcessor:
    def postprocess(self, decoded, output_type="pil"):
        assert output_type == "pil"
        return [f"img-{i}" for i in range(decoded.shape[0])]


class _DummyPipe:
    image_processor = _DummyImageProcessor()


class _IdentityView:
    def forward(self, decoded):
        return decoded


def _run(config: SampleConfig, monkeypatch):
    monkeypatch.setattr(sampler, "_sample_sd_like", lambda *args, **kwargs: torch.zeros((config.num_images, 4, 2, 2)))
    monkeypatch.setattr(sampler, "_decode_latents", lambda *args, **kwargs: torch.zeros((config.num_images, 3, 8, 8)))

    return sample_visual_anagram(
        pipe=_DummyPipe(),
        scheduler=None,
        views=[_IdentityView(), _IdentityView()],
        prompts=["a", "b"],
        conditioning=None,
        generator=torch.Generator(),
        config=config,
    )


def test_sample_visual_anagram_returns_single_image_shape(monkeypatch):
    image, view_images = _run(SampleConfig(num_images=1), monkeypatch)

    assert image == "img-0"
    assert view_images == ["img-0", "img-0"]


def test_sample_visual_anagram_returns_batched_images(monkeypatch):
    images, view_images = _run(SampleConfig(num_images=2), monkeypatch)

    assert images == ["img-0", "img-1"]
    assert view_images == [["img-0", "img-1"], ["img-0", "img-1"]]
