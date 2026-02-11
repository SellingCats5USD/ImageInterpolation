from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import DiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from tqdm.auto import tqdm

from .model import TextConditioning
from .views import View


@dataclass
class SampleConfig:
    width: int = 512
    height: int = 512
    steps: int = 50
    guidance_scale: float = 7.5




def _decode_latents(pipe: DiffusionPipeline, latents: torch.Tensor) -> torch.Tensor:
    latents_for_decode = latents / pipe.vae.config.scaling_factor

    needs_upcast = bool(getattr(pipe.vae.config, "force_upcast", False)) and hasattr(pipe, "upcast_vae")
    vae_restore_dtype: torch.dtype | None = None
    if needs_upcast:
        vae_restore_dtype = pipe.vae.dtype
        pipe.upcast_vae()

    # Keep latent dtype aligned with VAE weights for both upcast and non-upcast paths.
    latents_for_decode = latents_for_decode.to(pipe.vae.dtype)
    decoded = pipe.vae.decode(latents_for_decode, return_dict=False)[0]

    if needs_upcast and vae_restore_dtype is not None:
        pipe.vae.to(dtype=vae_restore_dtype)

    return decoded


@torch.no_grad()
def sample_visual_anagram(
    pipe: DiffusionPipeline,
    scheduler: DDIMScheduler,
    views: list[View],
    prompts: list[str],
    conditioning: TextConditioning,
    generator: torch.Generator,
    config: SampleConfig,
) -> tuple[Image.Image, list[Image.Image]]:
    if len(views) != len(prompts):
        raise ValueError("views and prompts must have the same length")

    device = pipe._execution_device
    dtype = pipe.unet.dtype

    scheduler.set_timesteps(config.steps, device=device)
    latent_shape = (
        1,
        pipe.unet.config.in_channels,
        config.height // pipe.vae_scale_factor,
        config.width // pipe.vae_scale_factor,
    )
    latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    for t in tqdm(scheduler.timesteps, desc="Denoising"):
        aligned_predictions = []
        for i, view in enumerate(views):
            latent_view = view.forward(latents)
            scaled_input = scheduler.scale_model_input(latent_view, t)

            unet_kwargs: dict[str, torch.Tensor] = {}
            if conditioning.add_time_ids is not None:
                unet_kwargs["added_cond_kwargs"] = {
                    "text_embeds": conditioning.pooled_text_embeddings[i : i + 1],
                    "time_ids": conditioning.add_time_ids[i : i + 1],
                }

            eps_uncond = pipe.unet(
                scaled_input,
                t,
                encoder_hidden_states=conditioning.uncond_embeddings[i : i + 1],
                **unet_kwargs,
            ).sample
            eps_text = pipe.unet(
                scaled_input,
                t,
                encoder_hidden_states=conditioning.text_embeddings[i : i + 1],
                **unet_kwargs,
            ).sample
            eps_cfg = eps_uncond + config.guidance_scale * (eps_text - eps_uncond)
            aligned_predictions.append(view.inverse(eps_cfg))

        eps_tilde = torch.stack(aligned_predictions, dim=0).mean(dim=0)
        latents = scheduler.step(eps_tilde, t, latents).prev_sample

    decoded = _decode_latents(pipe, latents)
    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    view_images: list[Image.Image] = []
    for view in views:
        transformed = view.forward(decoded)
        view_images.append(pipe.image_processor.postprocess(transformed, output_type="pil")[0])

    return image, view_images
