#!/usr/bin/env bash
set -euo pipefail

REPO_ID=${REPO_ID:-zjunlp/chatcell-large}
# Pin the current upstream revision so evaluations are reproducible.
REVISION=${REVISION:-3faf08011a3aa83955b8e808494ff765e0a7051c}
CHATCELL_ROOT=${CHATCELL_ROOT:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/chatcell}
MODEL_DIR=${MODEL_DIR:-${CHATCELL_ROOT}/chatcell-large}
PYTHON_BIN=${PYTHON_BIN:-python3}

mkdir -p "${CHATCELL_ROOT}" "${MODEL_DIR}"

# Keep both the Hub cache and the final snapshot on the large shared filesystem.
export HF_HOME=${HF_HOME:-${CHATCELL_ROOT}/.hf_home}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}

if ! "${PYTHON_BIN}" -c "import huggingface_hub" >/dev/null 2>&1; then
    DOWNLOAD_DEPS_DIR=${DOWNLOAD_DEPS_DIR:-${CHATCELL_ROOT}/.download_deps}
    mkdir -p "${DOWNLOAD_DEPS_DIR}"
    "${PYTHON_BIN}" -m pip install \
        --target "${DOWNLOAD_DEPS_DIR}" \
        "huggingface_hub>=0.23"
    export PYTHONPATH="${DOWNLOAD_DEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

"${PYTHON_BIN}" -c '
import sys
from huggingface_hub import snapshot_download

repo_id, revision, model_dir = sys.argv[1:4]
path = snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=model_dir,
    resume_download=True,
)
print(f"Downloaded {repo_id}@{revision} to {path}")
' "${REPO_ID}" "${REVISION}" "${MODEL_DIR}"

"${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
required = [
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
]
missing = [name for name in required if not (model_dir / name).is_file()]
if missing:
    raise SystemExit(f"Incomplete ChatCell snapshot; missing: {missing}")
config = json.loads((model_dir / "config.json").read_text())
if config.get("architectures") != ["T5ForConditionalGeneration"]:
    raise SystemExit("Unexpected architecture: " + repr(config.get("architectures")))
weight_gib = (model_dir / "pytorch_model.bin").stat().st_size / 1024**3
print(f"Verified T5ForConditionalGeneration snapshot ({weight_gib:.2f} GiB weights)")
' "${MODEL_DIR}"

printf 'ChatCell model path: %s\n' "${MODEL_DIR}"
