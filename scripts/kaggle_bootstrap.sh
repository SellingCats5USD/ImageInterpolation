#!/usr/bin/env bash
set -euo pipefail

# Bootstrap this repository inside a Kaggle notebook session so local VSCode and
# Kaggle GPU both use the exact same codebase/branch.
#
# Usage (inside Kaggle terminal or notebook cell via %%bash):
#   REPO_URL="https://github.com/<org>/<repo>.git" BRANCH="main" bash scripts/kaggle_bootstrap.sh
#
# Optional env vars:
#   WORKDIR        default: /kaggle/working
#   REPO_DIR_NAME  default: ImageInterpolation
#   HF_TOKEN       Hugging Face token for private/gated models

WORKDIR="${WORKDIR:-/kaggle/working}"
REPO_URL="${REPO_URL:-}"
BRANCH="${BRANCH:-main}"
REPO_DIR_NAME="${REPO_DIR_NAME:-ImageInterpolation}"

if [[ -z "${REPO_URL}" ]]; then
  echo "REPO_URL is required, e.g. REPO_URL=https://github.com/<org>/<repo>.git"
  exit 1
fi

cd "${WORKDIR}"

if [[ -d "${REPO_DIR_NAME}/.git" ]]; then
  echo "Repo already exists, syncing latest ${BRANCH}..."
  cd "${REPO_DIR_NAME}"
  git fetch --all --prune
  git checkout "${BRANCH}"
  git pull --ff-only origin "${BRANCH}"
else
  git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR_NAME}"
  cd "${REPO_DIR_NAME}"
fi

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ -n "${HF_TOKEN:-}" ]]; then
  python -m pip install huggingface_hub
  python - <<'PY'
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("Hugging Face login complete")
PY
fi

echo "Bootstrap complete."
echo "Next: source .venv/bin/activate && bash scripts/smoke_test.sh"
