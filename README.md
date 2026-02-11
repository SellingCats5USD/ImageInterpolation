# ImageInterpolation / Visual Anagrams Starter

This repository now contains a **single-GPU starter implementation** for recreating the core Visual Anagrams algorithm from Figure 2 of the paper:

1. apply view transform `v_i` to `x_t`
2. predict per-view noise `epsilon_t^i` conditioned on prompt `p_i`
3. invert each prediction with `v_i^-1`
4. average aligned predictions
5. run one reverse diffusion step

## What is included

- `src/views.py`: exact/invertible views (`identity`, `vflip`, `hflip`, `rot180`)
- `src/model.py`: SD model + tokenizer/text-encoder embedding setup
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
  --model runwayml/stable-diffusion-v1-5 \
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

## Notes

- Prefer `--dtype fp16` on GPU for lower VRAM.
- Start with `512x512`, then experiment with prompts and seeds.
- If one view dominates, reduce guidance or simplify prompts.
