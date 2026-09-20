# scOPD cell-annotation implementation

This directory contains the task-specific implementation of the three scOPD
stages: **Cold-Start Reasoning Fine-Tuning (CRFT)**, **Routed On-Policy
Self-Distillation (ROPSD)**, and **Disagreement-Gated Pairwise Refinement
(DGPR)**.

## Main code path

### 1. CRFT

- `prepare_reasoning_sft.py` builds completion-only reasoning examples.
- `prepare_nonthinking_reasoning_sft.py` converts trajectories to the explicit
  `<reasoning>...</reasoning>\n<answer>...</answer>` protocol.
- `sft_lora.py` performs LoRA fine-tuning and optional adapter merging.
- `cluster/submit_qwen3_8b_crft_rjob.sh` is the canonical launcher.

CRFT uses `enable_thinking=False`; the visible reasoning block is part of the
assistant completion and is supervised together with the final answer.

### 2. ROPSD

- `prepare_nonthinking_reasoning_rl_data.py` builds the full CellPuzzles RL
  split with the explicit response protocol.
- `prepare_o1_privileged_rl_data.py` attaches offline O1 trajectories used only
  as fallback privileged context.
- `reward_cello1.py::compute_scopd_ropsd_score` computes the dense reward,
  validates reasoning quality, and emits sibling/O1 routing metadata.
- `cluster/submit_qwen3_8b_ropsd_rjob.sh` is the canonical launcher.
- `../verl/trainer/config/scopd_ropsd.yaml` is the canonical configuration.

The route for every rollout is mutually exclusive:

1. qualified full-batch success -> GRPO;
2. unqualified rollout with a qualified sibling -> sibling DW-SDPO;
3. unqualified rollout without a sibling but with an O1 trace -> O1-fallback
   DW-SDPO;
4. otherwise -> fallback GRPO.

The trainer uses all rollout rewards to estimate group-relative advantages,
but only the GRPO-routed tokens consume those advantages. SDPO-routed tokens
use teacher-student generalized JSD instead. The final objective is globally
normalized by the total number of routed tokens.

### 3. DGPR

- `dgpr.py` implements output validation, Hamming-medoid selection, cell-wise
  support, stable/uncertain localization, challenger selection, and refinement
  acceptance.
- `infer_dgpr_vllm.py` runs multi-rollout generation and refinement with vLLM.
- `cluster/submit_qwen3_8b_dgpr_eval_rjob.sh` is the generic launcher.

DGPR is training-free and never uses evaluation labels to select or accept a
prediction.

## Default method settings

| Setting | Value |
| --- | ---: |
| ROPSD rollouts (`K`) | 8 |
| ROPSD epochs | 5 |
| Dense valid reward | `(cell accuracy + batch exact match) / 2` |
| Invalid reward | `-1` |
| EMA update rate | 0.05 |
| SDPO JSD alpha | 0.5 |
| Distillation support | top 100 logits + tail |
| Importance-ratio clip | 2.0 |
| Entropy coefficient (`beta`) | 1.0 |
| DGPR rollouts (`M`) | 8 |
| DGPR stable-cell threshold | 0.75 |

## Commands

From the repository root:

```bash
# Validate CRFT data generation and the explicit response protocol.
export PYTHONPATH=$PWD
python -m cell_annotation.prepare_nonthinking_reasoning_sft --help

# Inspect the canonical training jobs without submitting them.
REPO_DIR=$PWD DRY_RUN=1 \
  bash cell_annotation/cluster/submit_qwen3_8b_crft_rjob.sh
REPO_DIR=$PWD DRY_RUN=1 STUDENT_MODEL=/path/to/crft/merged_model \
  bash cell_annotation/cluster/submit_qwen3_8b_ropsd_rjob.sh

# Inspect a DGPR evaluation job.
REPO_DIR=$PWD DRY_RUN=1 MODEL_PATH=/path/to/ropsd/merged_model \
  bash cell_annotation/cluster/submit_qwen3_8b_dgpr_eval_rjob.sh
```

Historical launchers are preserved for reproducing baselines and ablations;
new users should start with the canonical commands above.

## Tests

```bash
export PYTHONPATH=$PWD
python -m pytest -q \
  cell_annotation/tests/test_reward_cello1.py \
  cell_annotation/tests/test_sibling_srpo_core.py \
  cell_annotation/tests/test_dgpr.py
```
