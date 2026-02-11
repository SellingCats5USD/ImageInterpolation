from src.model import resolve_model


def test_resolve_preset_sets_model_and_family() -> None:
    model_id, family = resolve_model(model=None, preset="sdxl", model_family="auto")
    assert model_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert family == "sdxl"


def test_resolve_custom_model_auto_family() -> None:
    model_id, family = resolve_model(model="foo/bar-xl", preset="none", model_family="auto")
    assert model_id == "foo/bar-xl"
    assert family == "sdxl"


def test_resolve_rejects_mismatched_family() -> None:
    try:
        resolve_model(model=None, preset="sdxl", model_family="sd15")
    except ValueError as exc:
        assert "expects model_family='sdxl'" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
