from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from diffusers import DDIMScheduler, DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline


@dataclass
class ModelBundle:
    pipe: DiffusionPipeline
    scheduler: DDIMScheduler
    model_family: str


@dataclass
class TextConditioning:
    uncond_embeddings: torch.Tensor
    text_embeddings: torch.Tensor
    uncond_pooled_embeddings: torch.Tensor | None = None
    pooled_text_embeddings: torch.Tensor | None = None
    add_time_ids: torch.Tensor | None = None


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "sd15": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "family": "sd15",
        "description": "Baseline SD 1.5 model (fast and broadly compatible).",
    },
    "dreamshaper8": {
        "model_id": "Lykon/dreamshaper-8",
        "family": "sd15",
        "description": "High-quality SD1.5 fine-tune with strong prompt following.",
    },
    "realisticvision60": {
        "model_id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        "family": "sd15",
        "description": "Modern SD1.5-based checkpoint tuned for realistic imagery.",
    },
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "family": "sdxl",
        "description": "SDXL base model with stronger composition and detail than SD1.5.",
    },
    "juggernautxl": {
        "model_id": "RunDiffusion/Juggernaut-XL-v9",
        "family": "sdxl",
        "description": "Popular SDXL fine-tune with strong aesthetics and fidelity.",
    },
}


def resolve_model(model: str | None, preset: str, model_family: str) -> tuple[str, str]:
    if preset != "none":
        selected = MODEL_PRESETS[preset]
        model = selected["model_id"]
        preset_family = selected["family"]
        if model_family == "auto":
            model_family = preset_family
        elif model_family != preset_family:
            raise ValueError(f"Preset '{preset}' expects model_family='{preset_family}'")

    if not model:
        raise ValueError("A model id is required when --preset none is used")

    if model_family == "auto":
        model_lc = model.lower()
        model_family = "sdxl" if "xl" in model_lc else "sd15"

    return model, model_family


def load_model(model_id: str, device: str, dtype: torch.dtype, model_family: str) -> ModelBundle:
    if model_family == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )
    elif model_family == "sd15":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return ModelBundle(pipe=pipe, scheduler=scheduler, model_family=model_family)


@torch.no_grad()
def build_text_embeddings(pipe: StableDiffusionPipeline, prompts: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_inputs = tokenizer(
        [""] * len(prompts),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_emb = text_encoder(text_inputs.input_ids.to(device))[0]
    uncond_emb = text_encoder(uncond_inputs.input_ids.to(device))[0]
    return uncond_emb, text_emb


@torch.no_grad()
def build_text_conditioning(
    pipe: DiffusionPipeline,
    prompts: list[str],
    device: str,
    model_family: str,
    width: int,
    height: int,
) -> TextConditioning:
    if model_family == "sd15":
        uncond_emb, text_emb = build_text_embeddings(pipe, prompts, device)
        return TextConditioning(uncond_embeddings=uncond_emb, text_embeddings=text_emb)

    if model_family == "sdxl":
        if not isinstance(pipe, StableDiffusionXLPipeline):
            raise TypeError("Expected a StableDiffusionXLPipeline for SDXL models")

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
            prompts,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=[""] * len(prompts),
        )
        add_time_ids = pipe._get_add_time_ids(
            original_size=(height, width),
            crops_coords_top_left=(0, 0),
            target_size=(height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
        ).to(device)
        add_time_ids = add_time_ids.repeat(len(prompts), 1)
        return TextConditioning(
            uncond_embeddings=negative_prompt_embeds,
            text_embeddings=prompt_embeds,
            uncond_pooled_embeddings=negative_pooled_prompt_embeds,
            pooled_text_embeddings=pooled_prompt_embeds,
            add_time_ids=add_time_ids,
        )

    raise ValueError(f"Unsupported model family: {model_family}")
