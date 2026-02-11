from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline


@dataclass
class ModelBundle:
    pipe: StableDiffusionPipeline
    scheduler: DDIMScheduler


def load_model(model_id: str, device: str, dtype: torch.dtype) -> ModelBundle:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return ModelBundle(pipe=pipe, scheduler=scheduler)


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
