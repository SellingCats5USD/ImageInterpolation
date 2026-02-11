import torch

from src.views import get_view


def test_all_views_are_invertible() -> None:
    x = torch.randn(2, 4, 8, 8)
    for name in ["identity", "vflip", "hflip", "rot180"]:
        view = get_view(name)
        restored = view.inverse(view.forward(x))
        assert torch.equal(restored, x)


def test_unknown_view_raises() -> None:
    try:
        get_view("bad_view")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
