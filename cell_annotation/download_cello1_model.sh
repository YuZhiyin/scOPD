#!/usr/bin/env bash
set -euo pipefail

REPO_ID=${REPO_ID:-ncbi/Cell-o1}
TARGET_DIR=${TARGET_DIR:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/cell-o1}

# The development machine can access Hugging Face only through the PJLab proxy.
unset https_proxy http_proxy
# shellcheck disable=SC1090
source <(curl -fsSL http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh)

# PJLab's HTTP proxy does not reliably support Xet CAS range reconstruction.
# Standard HTTP/LFS is slower but stable and supports local-dir resume.
export HF_HUB_DISABLE_XET=1

mkdir -p "${TARGET_DIR}"
export REPO_ID TARGET_DIR
python3 -c 'import os; from huggingface_hub import snapshot_download; print(snapshot_download(repo_id=os.environ["REPO_ID"], repo_type="model", local_dir=os.environ["TARGET_DIR"]))'

test -s "${TARGET_DIR}/config.json"
test -s "${TARGET_DIR}/tokenizer_config.json"
find "${TARGET_DIR}" -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) \
    -print -quit | grep -q .

echo "Downloaded ${REPO_ID} to ${TARGET_DIR}"
du -sh "${TARGET_DIR}"
