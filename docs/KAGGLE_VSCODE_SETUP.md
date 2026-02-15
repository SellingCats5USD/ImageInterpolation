# Kaggle GPU + VSCode Repo: Single-Codebase Workflow

This guide removes the "local code vs Kaggle code" split. You keep **one Git repo** (this repo), edit from VSCode, and run the same branch/commit on Kaggle GPU.

## Architecture (what changes)

- **Source of truth:** GitHub repo (`ImageInterpolation`).
- **Local VSCode:** where you edit, test, and commit.
- **Kaggle notebook GPU session:** pulls the same repo/branch and runs it.
- **No separate Kaggle-only notebook code path.**

## 0) One-time prerequisites

1. Push this repo to GitHub if it is not already there.
2. Install git locally and configure identity.
3. (Optional but recommended) create a Hugging Face token for model downloads.
4. If your GitHub repo is private, create a GitHub Personal Access Token (PAT).

## 1) Local (VSCode) routine

From your local machine:

```bash
git checkout -b feature/my-change
# edit code
bash scripts/smoke_test.sh
git add -A
git commit -m "Describe change"
git push -u origin feature/my-change
```

The key point: every Kaggle run should start from a pushed branch/commit.

## 2) Start Kaggle notebook using your repo

Create a new Kaggle notebook with GPU enabled.

In the first cell, clone and bootstrap the same repo/branch:

```bash
%%bash
cd /kaggle/working
git clone --branch feature/my-change https://github.com/<org>/<repo>.git ImageInterpolation
cd ImageInterpolation
REPO_URL="https://github.com/<org>/<repo>.git" \
  BRANCH="feature/my-change" \
  bash scripts/kaggle_bootstrap.sh
```

If you already cloned the repo in the session, you can rerun bootstrap without setting
`REPO_URL`; the script will reuse `origin` automatically:

```bash
%%bash
cd /kaggle/working/ImageInterpolation
BRANCH="feature/my-change" bash scripts/kaggle_bootstrap.sh
```

If your repo is private, use token-auth URL format:

```text
https://<github_username>:<github_pat>@github.com/<org>/<repo>.git
```

## 3) Add secrets safely in Kaggle

Use **Kaggle Add-ons → Secrets** (or notebook secrets UI) for:

- `HF_TOKEN` (for Hugging Face model access)
- optional GitHub token if cloning private repos

Then pass secrets to bootstrap:

```bash
%%bash
cd /kaggle/working/ImageInterpolation
export HF_TOKEN="$HF_TOKEN"
REPO_URL="https://github.com/<org>/<repo>.git" BRANCH="feature/my-change" bash scripts/kaggle_bootstrap.sh
```

## 4) Run training/inference on Kaggle GPU

```bash
%%bash
cd /kaggle/working/ImageInterpolation
source .venv/bin/activate
bash scripts/smoke_test.sh
python -m src.run \
  --preset sdxl \
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

## 5) Keep local + Kaggle in sync (day-to-day)

Use this loop:

1. Edit locally in VSCode.
2. Commit + push branch.
3. In Kaggle notebook, sync latest branch:

```bash
%%bash
cd /kaggle/working/ImageInterpolation
git fetch --all --prune
git checkout feature/my-change
git pull --ff-only origin feature/my-change
```

4. Re-run with GPU.

This ensures you always execute the exact code reviewed in Git.

## 6) Optional: work directly from Kaggle terminal + push back

If you make emergency fixes in Kaggle terminal:

```bash
cd /kaggle/working/ImageInterpolation
git config user.name "<your name>"
git config user.email "<your email>"
git add -A
git commit -m "Hotfix from Kaggle"
git push origin feature/my-change
```

Then pull locally in VSCode.

## 7) Connect VSCode to Kaggle Jupyter server (interactive notebooks)

If you want local VSCode notebook UI backed by Kaggle's running Jupyter server (GPU runtime), follow Kaggle's Jupyter server integration flow:

1. Launch a Kaggle notebook server session.
2. Obtain the server URL/token from Kaggle according to the official guide.
3. In VSCode Command Palette: **Jupyter: Specify Jupyter Server for Connections**.
4. Paste the Kaggle server URL/token.
5. Open `.ipynb` in VSCode; kernel should point to Kaggle-hosted runtime.

Reference: Kaggle Jupyter server docs: https://www.kaggle.com/docs/notebooks#kaggle-jupyter-server

> Practical recommendation: even when using remote Jupyter connection, keep `.py` source files in this repo and import them from notebooks. Do not duplicate logic in notebook-only cells.

## 8) Reproducibility guardrails

- Always log commit SHA in notebook outputs:

```bash
!cd /kaggle/working/ImageInterpolation && git rev-parse HEAD
```

- Pin dependencies via `requirements.txt`.
- Prefer branch names per experiment.
- Save artifacts under `outputs/` and upload/download as needed.

## 9) Troubleshooting

- **"Module not found" in Kaggle:** activate `.venv` and reinstall requirements.
- **Model download auth errors:** ensure `HF_TOKEN` secret is set and valid.
- **Detached branch in Kaggle:** `git checkout <branch>` then `git pull --ff-only`.
- **Out-of-memory:** lower resolution, switch to `--dtype fp16`, or use lighter preset.
