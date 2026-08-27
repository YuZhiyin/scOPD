#!/usr/bin/env bash
set -euo pipefail

# DEVELOPMENT-MACHINE ONLY.
# GPU rjobs are intentionally offline and never invoke this script.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
# shellcheck disable=SC1090
source <(curl -sSL http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh)

REPO_ID=${REPO_ID:-vandijklab/C2S-Pythia-410m-cell-type-prediction}
REVISION=${REVISION:-122c0ad022342f070549ebb37f3840fc08325cc0}
C2S_ROOT=${C2S_ROOT:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/C2S}
MODEL_DIR=${MODEL_DIR:-${C2S_ROOT}/C2S-Pythia-410m-cell-type-prediction}
PYTHON_BIN=${PYTHON_BIN:-python3}

mkdir -p "${C2S_ROOT}" "${MODEL_DIR}"
export HF_HOME=${HF_HOME:-${C2S_ROOT}/.hf_home}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
# The cluster proxy is substantially more reliable with plain HTTP range
# requests than with the Xet client, and range requests preserve .incomplete
# files across retries.
export HF_HUB_DISABLE_XET=1

if ! "${PYTHON_BIN}" -c "import huggingface_hub" >/dev/null 2>&1; then
    DOWNLOAD_DEPS_DIR=${DOWNLOAD_DEPS_DIR:-${C2S_ROOT}/.download_deps}
    mkdir -p "${DOWNLOAD_DEPS_DIR}"
    "${PYTHON_BIN}" -m pip install \
        --target "${DOWNLOAD_DEPS_DIR}" \
        "huggingface_hub>=0.23,<1.0"
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
    max_workers=1,
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "tokenizer*",
        "special_tokens_map.json",
        "README.md",
    ],
)
print(f"Downloaded {repo_id}@{revision} to {path}")
' "${REPO_ID}" "${REVISION}" "${MODEL_DIR}"

"${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
repo_id = sys.argv[2]
revision = sys.argv[3]
required = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]
missing = [name for name in required if not (model_dir / name).is_file()]
if missing:
    raise SystemExit(f"Incomplete C2S snapshot; missing: {missing}")
config = json.loads((model_dir / "config.json").read_text())
if config.get("architectures") != ["GPTNeoXForCausalLM"]:
    raise SystemExit("Unexpected architecture: " + repr(config.get("architectures")))
for forbidden in ("optimizer.pt", "rng_state.pth", "scheduler.pt"):
    if (model_dir / forbidden).exists():
        raise SystemExit(f"Training-only file should not be downloaded: {forbidden}")
weight_gib = (model_dir / "model.safetensors").stat().st_size / 1024**3
(model_dir / "offline_download_manifest.json").write_text(
    json.dumps(
        {
            "repo_id": repo_id,
            "revision": revision,
            "architecture": config.get("architectures"),
            "weight_bytes": (model_dir / "model.safetensors").stat().st_size,
        },
        indent=2,
    )
    + "\n"
)
print(f"Verified GPTNeoXForCausalLM snapshot ({weight_gib:.2f} GiB weights)")
' "${MODEL_DIR}" "${REPO_ID}" "${REVISION}"

printf 'Offline C2S model path: %s\n' "${MODEL_DIR}"
