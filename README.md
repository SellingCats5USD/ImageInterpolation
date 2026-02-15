# ImageInterpolation / Visual Anagrams Starter

This repository now contains a **single-GPU starter implementation** for recreating the core Visual Anagrams algorithm from Figure 2 of the paper:

1. apply view transform `v_i` to `x_t`
2. predict per-view noise `epsilon_t^i` conditioned on prompt `p_i`
3. invert each prediction with `v_i^-1`
4. average aligned predictions
5. run one reverse diffusion step

## What is included

- `src/views.py`: exact/invertible views (`identity`, `vflip`, `hflip`, `rot180`)
- `src/model.py`: SD/SDXL/FLUX model loading + tokenizer/text conditioning setup
- `src/sampler.py`: multi-view denoising loop with per-view CFG + aligned averaging
- `src/run.py`: CLI entrypoint for two-view anagrams
- `requirements.txt`: dependencies
- `tests/test_views.py`: basic invertibility tests for views
- `scripts/bootstrap.sh`: environment setup helper
- `scripts/smoke_test.sh`: quick non-model test

## Quick start

### 1) Create environment

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
```

### 2) Run a quick smoke test

```bash
bash scripts/smoke_test.sh
```

### 3) Generate an anagram (single GPU)

```bash
python -m src.run \
  --preset sd15 \
  --prompt_a "an oil painting of people around a campfire" \
  --prompt_b "an oil painting of an old man" \
  --view_a identity \
  --view_b vflip \
  --steps 50 \
  --guidance 7.5 \
  --seed 42 \
  --out outputs/campfire_oldman.png \
  --out_grid outputs/campfire_oldman_grid.png
```

### 4) Switch to stronger modern models

You can pick built-in presets without changing code:

```bash
# SDXL base 1.0 (modern architecture)
python -m src.run \
  --preset sdxl \
  --prompt_a "a detailed ukiyo-e style owl" \
  --prompt_b "a portrait photo of an astronaut" \
  --view_a identity \
  --view_b vflip
```

Available presets:

- `sd15`: `runwayml/stable-diffusion-v1-5` (baseline)
- `dreamshaper8`: `Lykon/dreamshaper-8` (strong SD1.5 fine-tune)
- `realisticvision60`: `SG161222/Realistic_Vision_V6.0_B1_noVAE` (modern realistic SD1.5)
- `sdxl`: `stabilityai/stable-diffusion-xl-base-1.0` (newer model family)
- `juggernautxl`: `RunDiffusion/Juggernaut-XL-v9` (popular SDXL fine-tune)
- `juggernautxl_lightning`: `RunDiffusion/Juggernaut-XL-Lightning` (optimized for ~4-8 steps)
- `flux1_schnell`: `black-forest-labs/FLUX.1-schnell` (fast FLUX model, usually 1-4 steps)

To use any custom model id directly:

```bash
python -m src.run --preset none --model <hf_model_id> --model_family auto ...
```

### Inference speed-ups

This repo now includes several built-in speed knobs:

- **Batched per-view UNet passes (default on)** via `--batch_unet` / `--no_batch_unet`
  - computes all view + CFG predictions in a single UNet forward per step for SD/SDXL models.
- **`--compile_unet`** to use `torch.compile` for repeated runs.
- **`--channels_last`** to use NHWC memory format on UNet (often faster on modern NVIDIA GPUs).
- **`--attention_slicing`** for lower VRAM (usually slower, but helpful if memory-bound).

Example fast run (Juggernaut XL Lightning):

```bash
python -m src.run \
  --preset juggernautxl_lightning \
  --steps 6 \
  --guidance 4.0 \
  --compile_unet \
  --channels_last \
  --prompt_a "a surreal owl made of stained glass" \
  --prompt_b "a cinematic portrait of a boxer"
```

Example FLUX.1 schnell run:

```bash
python -m src.run \
  --preset flux1_schnell \
  --model_family flux \
  --steps 4 \
  --guidance 1.0 \
  --prompt_a "a neon origami dragon" \
  --prompt_b "a minimalist ceramic teapot"
```

## Run this repo on Kaggle GPU (single codebase)

To avoid splitting code between local VSCode and Kaggle notebooks, use Git as the single source of truth and pull the same branch into Kaggle GPU sessions.

- Full setup guide: `docs/KAGGLE_VSCODE_SETUP.md`
- Kaggle helper bootstrap script: `scripts/kaggle_bootstrap.sh`

Minimal Kaggle cell:

```bash
%%bash
cd /kaggle/working
git clone --branch <branch> https://github.com/<org>/<repo>.git ImageInterpolation
cd ImageInterpolation
REPO_URL="https://github.com/<org>/<repo>.git" BRANCH="<branch>" bash scripts/kaggle_bootstrap.sh
```

For VSCode ↔ Kaggle Jupyter Server connection details, follow Kaggle docs: https://www.kaggle.com/docs/notebooks#kaggle-jupyter-server

## Notes

- Prefer `--dtype fp16` on GPU for lower VRAM.
- Start with `512x512`, then experiment with prompts and seeds.
- If one view dominates, reduce guidance or simplify prompts.
- SDXL typically benefits from larger resolutions and often lower guidance than SD1.5.

### Troubleshooting

- If Python prints `Error in sitecustomize ... ModuleNotFoundError: No module named 'wrapt'`, your environment is loading a `sitecustomize` hook that depends on `wrapt`. This repo now includes `wrapt` in `requirements.txt`; re-run `pip install -r requirements.txt` in your active environment.
