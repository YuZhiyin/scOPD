# scOPD

Official code and data release for **scOPD**, a three-stage framework for
reasoning-based cell type annotation:

1. **Reasoning SFT** teaches an explicit biological reasoning protocol.
2. **scOPD training** combines dense verifier rewards, sibling supervision,
   offline-o1 fallback supervision, GRPO/SDPO routing, and entropy-aware
   dynamic weighting.
3. **DGPR** is a training-free, consensus-triggered refinement procedure used
   at inference time.

This release contains the Qwen3/VERL implementation used for the main
experiments and ablations. The newer Qwen3.5/Slime implementation is
intentionally not included in this version.

## Method at a glance

For each prompt, the policy samples a group of `K` responses. The verifier
assigns the dense reward

```text
-1                                      invalid format/count/candidate set
(cell_accuracy + exact_batch_match) / 2 otherwise
```

and routes each response as follows:

```text
reasoning SFT checkpoint
          |
          v
   K on-policy responses
          |
          +-- exact response --------------------------> GRPO
          |
          +-- failed + qualified leave-one-out sibling -> sibling SDPO
          |
          +-- failed + offline o1 trace ---------------> o1 SDPO
          |
          +-- no eligible teacher ---------------------> GRPO fallback

SDPO token weights: exp(-beta * teacher_entropy), normalized per rollout batch
```

The default main configuration uses `K=8`, `beta=1.0`, five RL epochs, and a
4096-token response budget. DGPR uses `M=8` candidate rollouts and a consensus
threshold of `0.75` unless overridden.

## Repository layout

| Path | Contents |
| --- | --- |
| `cell_annotation/` | SFT, reward, evaluation, DGPR, data preparation, and experiment launchers |
| `cell_annotation/data/` | CellPuzzles raw data and the processed SFT/RL/evaluation data used by scOPD |
| `verl/` | SDPO/GRPO trainer with sibling routing, o1 fallback, and dynamic weighting |
| `verl/trainer/config/` | Main method and ablation configurations |
| `cell_annotation/tests/` | Reward, routing, SFT-tokenization, evaluation, and DGPR tests |
| `docs/EXPERIMENT_NOTES_ZH.md` | Detailed development and experiment notes in Chinese |
| `docs/UPSTREAM_SDPO_README.md` | README of the upstream SDPO codebase |

## Data

The included data are derived from the public
[`ncbi/CellPuzzles`](https://huggingface.co/datasets/ncbi/CellPuzzles)
dataset and the evaluation-only unseen splits used by Cell-o1. Important
counts are:

| Split | Rows |
| --- | ---: |
| CellPuzzles train | 6,912 |
| CellPuzzles test | 1,095 |
| Test-clean | 604 |
| Four unseen sets combined | 539 |
| Offline o1 reasoning traces | 3,912 |
| Reasoning-SFT train/dev | 3,521 / 391 |

The manifests under `cell_annotation/data/**/manifest.json` record source
URLs, checksums, split seeds, prompt modes, and row counts. Generated GenePT
embedding arrays and expanded C2S baseline caches are omitted because they are
large, reproducible intermediate artifacts and are not inputs to scOPD. See
[`cell_annotation/data/README.md`](cell_annotation/data/README.md) for the
complete inventory and data-license note.

To rebuild the canonical data from source:

```bash
export PYTHONPATH=$PWD
bash cell_annotation/scripts/download_and_prepare.sh
bash cell_annotation/scripts/prepare_o1_reasoning_sft_data.sh
python -m cell_annotation.prepare_nonthinking_reasoning_rl_data --help
python -m cell_annotation.prepare_o1_privileged_rl_data --help
```

## Installation

The trainer is based on
[`lasgroup/SDPO`](https://github.com/lasgroup/SDPO) and VERL. A CUDA machine is
required for model training and vLLM evaluation.

```bash
python -m pip install -e .
python -m pip install -r requirements-test.txt
```

The original experiments used Qwen3-8B and multi-GPU Ray jobs. Model weights
and training checkpoints are not stored in this repository.

## Reproducing the three stages

The launchers contain PJLab `rjob` defaults, but all important paths and
hyperparameters can be overridden through environment variables. Use
`DRY_RUN=1` to inspect the generated job before submission.

### 1. Reasoning SFT

```bash
REPO_DIR=$PWD \
BASE_MODEL=/path/to/Qwen3-8B \
DRY_RUN=1 \
bash cell_annotation/cluster/submit_qwen3_8b_nonthinking_reasoning_sft_rjob.sh
```

The reproduction split uses the 3,912 verified o1 traces, seed 0, and explicit
`<reasoning>...</reasoning><answer>...</answer>` targets while Qwen3 runs with
`enable_thinking=False`.

### 2. Dense sibling + o1 fallback + dynamic-weighted GRPO/SDPO

```bash
REPO_DIR=$PWD \
STUDENT_MODEL=/path/to/reasoning-sft/merged_model \
DRY_RUN=1 \
bash cell_annotation/cluster/submit_qwen3_8b_nonthinking_reasoning_dense_o1_dynamic_sibling_rjob.sh
```

The main trainer configuration is
[`nonthinking_reasoning_o1_fallback_dynamic_entropy_sibling_srpo.yaml`](verl/trainer/config/nonthinking_reasoning_o1_fallback_dynamic_entropy_sibling_srpo.yaml).
The adjacent launchers cover dense/sparse reward, no-SFT, no-o1, no-dynamic-
weighting, beta sensitivity, GRPO-only, and alternate joint-routing ablations.

### 3. Training-free DGPR

```bash
REPO_DIR=$PWD \
MODEL_PATH=/path/to/scopd/merged_model_step540 \
DRY_RUN=1 \
bash cell_annotation/cluster/submit_qwen3_8b_nt_dense_sibling_o1_dw_step540_dgpr_eval_rjob.sh
```

The implementation is in [`dgpr.py`](cell_annotation/dgpr.py) and
[`infer_dgpr_vllm.py`](cell_annotation/infer_dgpr_vllm.py). M-sensitivity
launchers for `M={2,4,6,8,10}` are included.

## Tests

CPU-only method tests can be run with:

```bash
pytest -q \
  cell_annotation/tests/test_reward_cello1.py \
  cell_annotation/tests/test_sibling_srpo_core.py \
  cell_annotation/tests/test_dgpr.py \
  tests/trainer/ppo/test_metric_utils_on_cpu.py
```

GPU integration tests require the same VERL/vLLM/Ray stack used for training.

## Attribution

This repository contains modifications to the Apache-2.0-licensed SDPO/VERL
codebase. See [`NOTICE`](NOTICE) and the upstream README for attribution.
CellPuzzles and other third-party data remain subject to their source terms;
the repository's Apache-2.0 license applies to code, not third-party datasets.

## License

Code is released under the [Apache License 2.0](LICENSE).
