from __future__ import annotations

from types import SimpleNamespace

import src.run as run_mod


class _DummyTorch:
    float16 = object()
    float32 = object()

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return True


def test_main_avoids_duplicate_identity_in_grid(monkeypatch) -> None:
    monkeypatch.setattr(run_mod, "torch", _DummyTorch)

    args = SimpleNamespace(
        preset="sd15",
        model=None,
        model_family="auto",
        prompt_a="a",
        prompt_b="b",
        view_a="identity",
        view_b="vflip",
        steps=1,
        guidance=1.0,
        width=64,
        height=64,
        seed=1,
        device="cuda",
        dtype="fp16",
        out="out.png",
        out_grid="grid.png",
        batch_unet=True,
        attention_slicing=False,
        channels_last=False,
        compile_unet=False,
    )
    monkeypatch.setattr(run_mod, "parse_args", lambda: args)
    monkeypatch.setattr(run_mod, "resolve_model", lambda **_: ("model", "sd15"))
    monkeypatch.setattr(run_mod, "load_model", lambda **_: SimpleNamespace(pipe=SimpleNamespace(), scheduler=object(), model_family="sd15"))
    monkeypatch.setattr(run_mod, "build_text_conditioning", lambda **_: object())
    monkeypatch.setattr(run_mod, "make_generator", lambda *_: object())
    monkeypatch.setattr(run_mod, "get_view", lambda name: name)

    image = object()
    transformed_views = [image, object()]
    monkeypatch.setattr(run_mod, "sample_visual_anagram", lambda **_: (image, transformed_views))

    captured = {}

    def _capture_grid(images):
        captured["images"] = images
        return "grid"

    monkeypatch.setattr(run_mod, "make_horizontal_grid", _capture_grid)
    monkeypatch.setattr(run_mod, "save_pil", lambda *_: None)

    run_mod.main()

    assert captured["images"] == [image, transformed_views[1]]
