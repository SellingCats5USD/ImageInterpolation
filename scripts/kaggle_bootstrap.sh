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

is_kaggle() {
  [[ -n "${KAGGLE_URL_BASE:-}" ]] || [[ -d /kaggle ]]
}

validate_repo_url() {
  local repo_url="$1"
  if [[ "${repo_url}" == *"<org>"* ]] || [[ "${repo_url}" == *"<repo>"* ]]; then
    echo "REPO_URL still contains placeholders. Replace <org>/<repo>, e.g. REPO_URL=https://github.com/my-org/ImageInterpolation.git"
    exit 1
  fi
}

cd "${WORKDIR}"

if [[ -d "${REPO_DIR_NAME}/.git" ]]; then
  if [[ -z "${REPO_URL}" ]]; then
    REPO_URL="$(git -C "${REPO_DIR_NAME}" remote get-url origin 2>/dev/null || true)"
  fi

  if [[ -z "${REPO_URL}" ]]; then
    echo "REPO_URL is required when ${REPO_DIR_NAME} has no origin remote configured."
    exit 1
  fi
  validate_repo_url "${REPO_URL}"

  echo "Repo already exists, syncing latest ${BRANCH}..."
  cd "${REPO_DIR_NAME}"
  git remote set-url origin "${REPO_URL}"
  git fetch --all --prune
  git checkout "${BRANCH}"
  git pull --ff-only origin "${BRANCH}"
else
  if [[ -z "${REPO_URL}" ]]; then
    echo "REPO_URL is required, e.g. REPO_URL=https://github.com/my-org/ImageInterpolation.git"
    exit 1
  fi
  validate_repo_url "${REPO_URL}"

  git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR_NAME}"
  cd "${REPO_DIR_NAME}"
fi

create_venv() {
  local venv_dir="$1"
  local use_system_site_packages=0

  if is_kaggle; then
    # Kaggle images already ship heavy ML packages (like torch). Reuse them.
    use_system_site_packages=1
  fi

  # If a previous run left behind a broken venv, recreate it.
  if [[ -d "${venv_dir}" ]] && ! "${venv_dir}/bin/python" -m pip --version >/dev/null 2>&1; then
    rm -rf "${venv_dir}"
  fi

  # If this venv was created without system site-packages on Kaggle, recreate it.
  if [[ -d "${venv_dir}" ]] && [[ "${use_system_site_packages}" -eq 1 ]]; then
    if [[ ! -f "${venv_dir}/pyvenv.cfg" ]] || ! grep -Eq '^include-system-site-packages = true$' "${venv_dir}/pyvenv.cfg"; then
      echo "Recreating ${venv_dir} with --system-site-packages for Kaggle..."
      rm -rf "${venv_dir}"
    fi
  fi

  if [[ ! -d "${venv_dir}" ]]; then
    local -a venv_args=()
    if [[ "${use_system_site_packages}" -eq 1 ]]; then
      venv_args+=(--system-site-packages)
    fi

    if ! python -m venv "${venv_args[@]}" "${venv_dir}"; then
      echo "python -m venv failed; falling back to virtualenv bootstrap..."
      python -m pip install --user --upgrade virtualenv
      if [[ "${use_system_site_packages}" -eq 1 ]]; then
        python -m virtualenv --system-site-packages "${venv_dir}"
      else
        python -m virtualenv "${venv_dir}"
      fi
    fi
  fi
}

create_venv .venv
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
