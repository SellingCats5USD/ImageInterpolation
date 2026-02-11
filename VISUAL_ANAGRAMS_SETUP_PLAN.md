# Visual Anagrams (Single-GPU) Setup Plan

This plan is for reproducing **Visual Anagrams** (multi-view denoising) on **1 GPU** using a **pre-trained diffusion model** and the Hugging Face `diffusers` stack.

## 1) Target baseline and constraints

- **Goal:** Recreate the paper’s core algorithm (Figure 2):
  1. apply view `v_i` to latent/noisy image `x_t`
  2. predict noise `ε_t^i` conditioned on prompt `p_i`
  3. invert view `v_i^-1` on each prediction
  4. average aligned predictions to get `\tilde{ε}_t`
  5. run one reverse diffusion step with `\tilde{ε}_t`
- **Hardware:** 1 GPU, speed not critical.
- **Practical first scope:** 2-view anagrams (`identity`, `vertical_flip`) then extend to 3–4 views.

## 2) Recommended model choice (pre-trained, 1 GPU friendly)

Use **Stable Diffusion v1.5** (`runwayml/stable-diffusion-v1-5`) with DDIM scheduler:

- widely available and stable in `diffusers`
- manageable VRAM with FP16 (typically ~8–12 GB depending on resolution/batch)
- easiest path for custom denoising loop (manual UNet calls)

Optional alternatives after baseline:
- SD 2.1 base (`stabilityai/stable-diffusion-2-1-base`)
- SDXL base only if you have enough VRAM and want higher quality

## 3) Environment setup

## 3.1 System prerequisites

- NVIDIA GPU + current driver
- CUDA-compatible PyTorch install
- Python 3.10+ recommended

## 3.2 Python dependencies

Install:
- `torch`, `torchvision`, `xformers` (optional but useful)
- `diffusers`, `transformers`, `accelerate`, `safetensors`
- `Pillow`, `numpy`, `matplotlib`, `einops`, `tqdm`

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors pillow numpy matplotlib einops tqdm xformers
```

## 3.3 Runtime sanity checks

- verify `torch.cuda.is_available()` is true
- load the SD pipeline once
- generate a standard image from one prompt before implementing multi-view logic

## 4) Project structure

```text
visual-anagrams/
  configs/
    base.yaml
  src/
    views.py              # view transforms + inverse transforms
    prompts.py            # prompt pair/group definitions
    model.py              # model loading, text embeddings, scheduler setup
    sampler.py            # core multi-view denoising loop
    run.py                # CLI entrypoint
    utils.py              # seed control, image save/grid, logging
  outputs/
  requirements.txt
```

## 5) Core algorithm implementation plan

## 5.1 Represent views

Implement each `view` as:
- `forward(x)` → transformed latent/image
- `inverse(x)` → inverse transform

Start with discrete transforms that are exact/invertible:
- `identity`
- `vflip` (up/down)
- `hflip` (left/right)
- `rot180`

(Leave arbitrary-angle rotations for later; interpolation can blur/inject artifacts.)

## 5.2 Operate in latent space (recommended)

Use SD’s latent denoising loop directly:
- initialize latent noise `x_T ~ N(0, I)`
- for each timestep `t`, compute aggregated noise prediction using multiple views
- call scheduler step once using aggregated noise

This mirrors the paper’s iterative denoising while avoiding repeated VAE encode/decode every step.

## 5.3 Per-step multi-view noise aggregation

At each timestep `t`:

1. For each view `v_i` with prompt `p_i`:
   - `x_t_i = v_i.forward(x_t)`
   - UNet noise prediction with prompt embedding `e_i`: `ε_t^i = UNet(x_t_i, t, e_i)`
   - classifier-free guidance per-view (using matching unconditional embedding)
   - align back: `ε̂_t^i = v_i.inverse(ε_t^i)`

2. Aggregate aligned estimates:
   - `\tilde{ε}_t = mean_i(ε̂_t^i)`
   - (optionally weighted mean later)

3. Reverse step once:
   - `x_{t-1} = scheduler.step(\tilde{ε}_t, t, x_t).prev_sample`

## 5.4 CFG detail (important)

For each view/prompt pair, compute:

- `ε_uncond_i` from unconditional embedding
- `ε_text_i` from prompt embedding
- `ε_cfg_i = ε_uncond_i + s * (ε_text_i - ε_uncond_i)`

Then invert + average `ε_cfg_i` across views.

## 5.5 Minimal pseudocode

```python
x_t = randn_latents(seed)
for t in scheduler.timesteps:
    eps_aligned = []
    for view, emb_uncond, emb_text in views_and_embeddings:
        x_view = view.forward(x_t)
        eps_u = unet(x_view, t, encoder_hidden_states=emb_uncond).sample
        eps_c = unet(x_view, t, encoder_hidden_states=emb_text).sample
        eps_cfg = eps_u + guidance_scale * (eps_c - eps_u)
        eps_aligned.append(view.inverse(eps_cfg))

    eps_tilde = torch.stack(eps_aligned, dim=0).mean(dim=0)
    x_t = scheduler.step(eps_tilde, t, x_t).prev_sample

image = vae_decode(x_t)
```

## 6) Initial experiment matrix

Start small and controlled:

- Resolution: `512x512`
- Steps: `50` DDIM steps
- Guidance scale: `6.5`, `7.5`, `8.5`
- Prompts (2-view):
  - view1 (`identity`): “an oil painting of people around a campfire”
  - view2 (`vflip`): “an oil painting of an old man”
- Seeds: test 8–16 seeds and save grids

Choose best seed by subjective illusion quality.

## 7) Quality-improving knobs (after baseline works)

- **Prompt balancing:** tune adjective/detail levels so one view does not dominate.
- **Weighted averaging:** `\tilde{ε}_t = Σ w_i ε̂_t^i` if one view is weak.
- **Timestep-dependent weights:** e.g., stronger averaging at early timesteps.
- **Negative prompts:** suppress unwanted motifs.
- **Scheduler variants:** try Euler/DPMSolver after DDIM baseline.

## 8) Common pitfalls and fixes

- **One prompt always wins:** lower CFG, simplify dominant prompt, weight weaker prompt higher.
- **No illusion after transform:** ensure exact inverse transform and consistent tensor dims.
- **Blurry/muddy outputs:** reduce step count extremes, tune CFG toward 6–8, test another seed.
- **OOM on 1 GPU:** use FP16, attention slicing, smaller batch (=1), disable extra logging tensors.

## 9) Deliverables checklist

- [ ] CLI that accepts `--prompt_a`, `--prompt_b`, `--view_b`, `--seed`
- [ ] Deterministic runs via fixed seeds
- [ ] Saved outputs:
  - original image
  - transformed view image(s)
  - side-by-side grid
- [ ] Config file for reproducible default run
- [ ] README with exact run commands

## 10) Suggested milestone schedule

1. **Day 1:** environment + single-prompt SD sanity run
2. **Day 2:** implement view transforms and inverse checks
3. **Day 3:** implement multi-view denoising loop (2 views)
4. **Day 4:** seed sweep + prompt tuning + documentation

## 11) First command to target (once implemented)

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
  --out outputs/campfire_oldman_seed42.png
```

This is the lowest-risk path to reproduce the paper’s core idea on a single GPU without training a model from scratch.
