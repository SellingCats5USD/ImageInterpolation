from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import MODEL_PRESETS, build_text_conditioning, load_model, resolve_model
from .sampler import SampleConfig, sample_visual_anagram
from .utils import make_generator, make_horizontal_grid, save_pil
from .views import get_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Visual Anagrams via multi-view denoising")
    parser.add_argument("--preset", choices=list(MODEL_PRESETS.keys()) + ["none"], default="sd15")
    parser.add_argument("--model", default=None, help="Override model id directly (use with --preset none).")
    parser.add_argument("--model_family", choices=["auto", "sd15", "sdxl", "flux"], default="auto")
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
    parser.add_argument("--batch_unet", dest="batch_unet", action="store_true", default=True, help="Batch per-view + CFG UNet calls into one forward pass (faster).")
    parser.add_argument("--no_batch_unet", dest="batch_unet", action="store_false", help="Disable batched UNet pass.")
    parser.add_argument("--attention_slicing", action="store_true", help="Lower VRAM usage at some cost to speed.")
    parser.add_argument("--channels_last", action="store_true", help="Use channels-last memory format for UNet.")
    parser.add_argument("--compile_unet", action="store_true", help="Use torch.compile on UNet for faster repeated inference.")
    parser.add_argument("--out", default="outputs/anagram.png")
    parser.add_argument("--out_grid", default="outputs/anagram_grid.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("--num_images must be >= 1")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model_id, model_family = resolve_model(model=args.model, preset=args.preset, model_family=args.model_family)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    bundle = load_model(
        model_id=model_id,
        device=args.device,
        dtype=dtype,
        model_family=model_family,
        enable_attention_slicing=args.attention_slicing,
        channels_last=args.channels_last,
        compile_unet=args.compile_unet,
    )

    prompts = [args.prompt_a, args.prompt_b]
    views = [get_view(args.view_a), get_view(args.view_b)]

    conditioning = build_text_conditioning(
        pipe=bundle.pipe,
        prompts=prompts,
        device=args.device,
        model_family=bundle.model_family,
        width=args.width,
        height=args.height,
    )
    generator = make_generator(args.seed, args.device)

    config = SampleConfig(
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance,
        batch_unet=args.batch_unet,
    )

    image, transformed_views = sample_visual_anagram(
        pipe=bundle.pipe,
        scheduler=bundle.scheduler,
        views=views,
        prompts=prompts,
        conditioning=conditioning,
        generator=generator,
        config=config,
    )

    if args.num_images == 1:
        save_pil(image, args.out)
        # `transformed_views` already includes the first view result; when view_a is identity
        # that is the same orientation as `image`. Keep `image` and append only additional
        # transformed views to avoid showing the normal orientation twice in the grid.
        grid = make_horizontal_grid([image, *transformed_views[1:]])
        save_pil(grid, args.out_grid)

        print(f"Saved base image to {args.out}")
        print(f"Saved comparison grid to {args.out_grid}")
    else:
        out_path = Path(args.out)
        out_grid_path = Path(args.out_grid)
        for i, base_image in enumerate(image):
            base_path = out_path.with_name(f"{out_path.stem}_{i:03d}{out_path.suffix}")
            grid_path = out_grid_path.with_name(f"{out_grid_path.stem}_{i:03d}{out_grid_path.suffix}")

            transformed_for_image = [view_images[i] for view_images in transformed_views]
            save_pil(base_image, str(base_path))
            grid = make_horizontal_grid([base_image, *transformed_for_image[1:]])
            save_pil(grid, str(grid_path))

            print(f"[{i + 1}/{args.num_images}] Saved base image to {base_path}")
            print(f"[{i + 1}/{args.num_images}] Saved comparison grid to {grid_path}")
    print(f"Model: {model_id} ({bundle.model_family})")


if __name__ == "__main__":
    main()
