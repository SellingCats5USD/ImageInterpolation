import torch

from src.sampler import _decode_latents


class _DummyConv:
    def __init__(self, dtype: torch.dtype):
        self.weight = torch.zeros(1, dtype=dtype)


class _DummyVAE:
    def __init__(self, force_upcast: bool):
        self.config = type("Cfg", (), {"scaling_factor": 0.13025, "force_upcast": force_upcast})()
        self.dtype = torch.float16
        self.device = torch.device("cpu")
        self.post_quant_conv = _DummyConv(torch.float16)
        self.to_calls = []
        self.last_decode_dtype = None

    def decode(self, latents, return_dict=False):
        self.last_decode_dtype = latents.dtype
        return (latents,)

    def to(self, dtype):
        self.to_calls.append(dtype)
        self.dtype = dtype
        self.post_quant_conv = _DummyConv(dtype)
        return self


class _DummyPipe:
    def __init__(self, force_upcast: bool):
        self.vae = _DummyVAE(force_upcast)
        self.upcast_calls = 0

    def upcast_vae(self):
        self.upcast_calls += 1
        self.vae.dtype = torch.float32
        self.vae.post_quant_conv = _DummyConv(torch.float32)


def test_decode_latents_upcasts_and_restores_for_force_upcast_vae() -> None:
    pipe = _DummyPipe(force_upcast=True)
    latents = torch.ones((1, 4, 2, 2), dtype=torch.float16)

    decoded = _decode_latents(pipe, latents)

    assert pipe.upcast_calls == 1
    assert pipe.vae.to_calls == [torch.float16]
    assert decoded.dtype == torch.float32


def test_decode_latents_without_upcast_keeps_dtype() -> None:
    pipe = _DummyPipe(force_upcast=False)
    latents = torch.ones((1, 4, 2, 2), dtype=torch.float16)

    decoded = _decode_latents(pipe, latents)

    assert pipe.upcast_calls == 0
    assert pipe.vae.to_calls == []
    assert decoded.dtype == torch.float16


def test_decode_latents_casts_input_to_vae_dtype_without_upcast() -> None:
    pipe = _DummyPipe(force_upcast=False)
    latents = torch.ones((1, 4, 2, 2), dtype=torch.float32)

    _decode_latents(pipe, latents)

    assert pipe.vae.last_decode_dtype == torch.float16
