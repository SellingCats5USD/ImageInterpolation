from __future__ import annotations

import argparse

import torch

from .model import build_text_embeddings, load_model
from .sampler import SampleConfig, sample_visual_anagram
from .utils import make_generator, make_horizontal_grid, save_pil
from .views import get_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Visual Anagrams via multi-view denoising")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt_a", required=True)
    parser.add_argument("--prompt_b", required=True)
    parser.add_argument("--view_a", default="identity")
    parser.add_argument("--view_b", default="vflip")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--out", default="outputs/anagram.png")
    parser.add_argument("--out_grid", default="outputs/anagram_grid.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    bundle = load_model(model_id=args.model, device=args.device, dtype=dtype)

    prompts = [args.prompt_a, args.prompt_b]
    views = [get_view(args.view_a), get_view(args.view_b)]

    uncond_emb, text_emb = build_text_embeddings(bundle.pipe, prompts, args.device)
    generator = make_generator(args.seed, args.device)

    config = SampleConfig(
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance,
    )

    image, transformed_views = sample_visual_anagram(
        pipe=bundle.pipe,
        scheduler=bundle.scheduler,
        views=views,
        prompts=prompts,
        uncond_embeddings=uncond_emb,
        text_embeddings=text_emb,
        generator=generator,
        config=config,
    )

    save_pil(image, args.out)
    grid = make_horizontal_grid([image, *transformed_views])
    save_pil(grid, args.out_grid)

    print(f"Saved base image to {args.out}")
    print(f"Saved comparison grid to {args.out_grid}")


if __name__ == "__main__":
    main()
