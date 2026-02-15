from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import DiffusionPipeline, FluxPipeline
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
    num_images: int = 1
    batch_unet: bool = True


def _decode_latents(pipe: DiffusionPipeline, latents: torch.Tensor) -> torch.Tensor:
    latents_for_decode = latents / pipe.vae.config.scaling_factor

    needs_upcast = bool(getattr(pipe.vae.config, "force_upcast", False)) and hasattr(pipe, "upcast_vae")
    vae_restore_dtype: torch.dtype | None = None
    if needs_upcast:
        vae_restore_dtype = pipe.vae.dtype
        pipe.upcast_vae()
        latents_for_decode = latents_for_decode.to(pipe.vae.dtype)

    decoded = pipe.vae.decode(latents_for_decode, return_dict=False)[0]

    if needs_upcast and vae_restore_dtype is not None:
        pipe.vae.to(dtype=vae_restore_dtype)

    return decoded


def _sample_sd_like(
    pipe: DiffusionPipeline,
    scheduler: DDIMScheduler,
    views: list[View],
    conditioning: TextConditioning,
    generator: torch.Generator,
    config: SampleConfig,
) -> torch.Tensor:
    device = pipe._execution_device
    dtype = pipe.unet.dtype
    batch_size = config.num_images
    num_views = len(views)

    scheduler.set_timesteps(config.steps, device=device)
    latent_shape = (
        batch_size,
        pipe.unet.config.in_channels,
        config.height // pipe.vae_scale_factor,
        config.width // pipe.vae_scale_factor,
    )
    latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    for t in tqdm(scheduler.timesteps, desc="Denoising"):
        latent_views = torch.cat([view.forward(latents) for view in views], dim=0)
        scaled_views = scheduler.scale_model_input(latent_views, t)

        if config.batch_unet:
            model_input = torch.cat([scaled_views, scaled_views], dim=0)
            encoder_hidden_states = torch.cat([conditioning.uncond_embeddings, conditioning.text_embeddings], dim=0)
            unet_kwargs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
            if conditioning.add_time_ids is not None:
                pooled = torch.cat([conditioning.uncond_pooled_embeddings, conditioning.pooled_text_embeddings], dim=0)
                time_ids = torch.cat([conditioning.add_time_ids, conditioning.add_time_ids], dim=0)
                unet_kwargs["added_cond_kwargs"] = {"text_embeds": pooled, "time_ids": time_ids}

            eps = pipe.unet(model_input, t, encoder_hidden_states=encoder_hidden_states, **unet_kwargs).sample
            eps_uncond, eps_text = torch.chunk(eps, 2, dim=0)
        else:
            eps_uncond_list = []
            eps_text_list = []
            for i in range(len(views)):
                unet_kwargs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
                if conditioning.add_time_ids is not None:
                    unet_kwargs["added_cond_kwargs"] = {
                        "text_embeds": conditioning.pooled_text_embeddings[i : i + 1],
                        "time_ids": conditioning.add_time_ids[i : i + 1],
                    }

                eps_uncond_list.append(
                    pipe.unet(
                        scaled_views[i : i + 1],
                        t,
                        encoder_hidden_states=conditioning.uncond_embeddings[i : i + 1],
                        **unet_kwargs,
                    ).sample
                )
                eps_text_list.append(
                    pipe.unet(
                        scaled_views[i : i + 1],
                        t,
                        encoder_hidden_states=conditioning.text_embeddings[i : i + 1],
                        **unet_kwargs,
                    ).sample
                )
            eps_uncond = torch.cat(eps_uncond_list, dim=0)
            eps_text = torch.cat(eps_text_list, dim=0)

        eps_cfg = eps_uncond + config.guidance_scale * (eps_text - eps_uncond)
        aligned_predictions = [view.inverse(eps_cfg[i : i + 1]) for i, view in enumerate(views)]
        eps_tilde = torch.stack(aligned_predictions, dim=0).mean(dim=0)
        latents = scheduler.step(eps_tilde, t, latents).prev_sample

    return latents


def _sample_flux(
    pipe: FluxPipeline,
    scheduler,
    views: list[View],
    conditioning: TextConditioning,
    generator: torch.Generator,
    config: SampleConfig,
) -> torch.Tensor:
    device = pipe._execution_device
    dtype = pipe.transformer.dtype

    latents, latent_image_ids = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.transformer.config.in_channels // 4,
        height=config.height,
        width=config.width,
        dtype=dtype,
        device=device,
        generator=generator,
    )

    timesteps, _ = pipe.retrieve_timesteps(scheduler, num_inference_steps=config.steps, device=device, sigmas=None)
    guidance = torch.full([1], config.guidance_scale, device=device, dtype=torch.float32)

    for t in tqdm(timesteps, desc="Denoising"):
        latents_unpacked = pipe._unpack_latents(latents, config.height, config.width, pipe.vae_scale_factor)
        view_latents = [view.forward(latents_unpacked) for view in views]
        packed_views = torch.cat(
            [pipe._pack_latents(v, 1, v.shape[1], v.shape[2], v.shape[3]) for v in view_latents],
            dim=0,
        )

        timestep = t.expand(packed_views.shape[0]).to(packed_views.dtype)
        noise_pred = pipe.transformer(
            hidden_states=packed_views,
            timestep=timestep / 1000,
            guidance=guidance.expand(packed_views.shape[0]),
            pooled_projections=conditioning.pooled_text_embeddings,
            encoder_hidden_states=conditioning.text_embeddings,
            txt_ids=conditioning.text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=pipe.joint_attention_kwargs,
            return_dict=False,
        )[0]

        unpacked_noise = [
            pipe._unpack_latents(noise_pred[i : i + 1], config.height, config.width, pipe.vae_scale_factor)
            for i in range(len(views))
        ]
        aligned = [view.inverse(pred) for view, pred in zip(views, unpacked_noise)]
        packed_aligned = torch.cat(
            [pipe._pack_latents(x, 1, x.shape[1], x.shape[2], x.shape[3]) for x in aligned],
            dim=0,
        )
        noise_mean = packed_aligned.mean(dim=0, keepdim=True)
        latents = scheduler.step(noise_mean, t, latents, return_dict=False)[0]

    return latents


@torch.no_grad()
def sample_visual_anagram(
    pipe: DiffusionPipeline,
    scheduler,
    views: list[View],
    prompts: list[str],
    conditioning: TextConditioning,
    generator: torch.Generator,
    config: SampleConfig,
) -> tuple[Image.Image, list[Image.Image]] | tuple[list[Image.Image], list[list[Image.Image]]]:
    if len(views) != len(prompts):
        raise ValueError("views and prompts must have the same length")

    if isinstance(pipe, FluxPipeline):
        latents = _sample_flux(pipe, scheduler, views, conditioning, generator, config)
        latents = pipe._unpack_latents(latents, config.height, config.width, pipe.vae_scale_factor)
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        decoded = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        view_images = [pipe.image_processor.postprocess(view.forward(decoded), output_type="pil")[0] for view in views]
        return image, view_images

    latents = _sample_sd_like(pipe, scheduler, views, conditioning, generator, config)
    decoded = _decode_latents(pipe, latents)
    images = pipe.image_processor.postprocess(decoded, output_type="pil")
    view_images = [pipe.image_processor.postprocess(view.forward(decoded), output_type="pil") for view in views]

    if config.num_images == 1:
        return images[0], [imgs[0] for imgs in view_images]

    return images, view_images
