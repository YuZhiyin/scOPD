# scOPD

Official implementation of **scOPD: Routed On-Policy Self-Distillation for
Large Language Model-Based Single-Cell Annotation**.

scOPD uses complementary signals from multiple model rollouts: successful
sibling trajectories provide fine-grained reasoning supervision during
training, while disagreement among valid rollouts guides conservative
refinement during inference.

## Framework

scOPD consists of three stages:

1. **Cold-Start Reasoning Fine-Tuning (CRFT)** distills biological reasoning
   trajectories into an explicit response protocol.
2. **Routed On-Policy Self-Distillation (ROPSD)** routes each on-policy
   rollout to outcome-based GRPO or token-level self-distillation according to
   rollout quality and the availability of privileged reasoning.
3. **Disagreement-Gated Pairwise Refinement (DGPR)** uses rollout
   disagreement to localize uncertain cells and performs constraint-aware
   refinement with a medoid anchor and a supported challenger.

The model uses non-thinking inference and exposes its reasoning through the
following supervised protocol:

```text
<reasoning>
Biological reasoning grounded in marker genes, cell context, and the
one-to-one matching constraint.
</reasoning>
<answer>
cell type 1 | cell type 2 | ...
</answer>
```

## ROPSD at a glance

For each annotation task, the policy samples `K` on-policy rollouts. A
structurally valid response receives

```text
(cell-level accuracy + batch exact match) / 2
```

and an invalid response receives `-1`. A rollout is *qualified* only when its
entire batch assignment is correct, its output is valid, and its reasoning
passes the non-degeneration guard. ROPSD then applies the following mutually
exclusive routing:

```text
K on-policy rollouts
        |
        +-- qualified ---------------------------------> GRPO
        |
        +-- unqualified + qualified sibling ----------> sibling DW-SDPO
        |
        +-- unqualified + no sibling + O1 trajectory -> O1-fallback DW-SDPO
        |
        +-- no eligible privileged context -----------> fallback GRPO
```

The self-teacher is an exponential-moving-average copy of the policy. It sees
the same student trajectory plus privileged sibling or O1 reasoning and
re-scores the student's next-token distributions; it does not generate a new
trajectory. SDPO uses generalized Jensen--Shannon divergence (`alpha=0.5`),
top-100 logits plus a tail bucket, and clipped importance weights. Its
entropy-aware token weights are
`exp(-beta * teacher_entropy)` and are normalized across all routed SDPO
tokens in the PPO mini-batch and data-parallel group. GRPO and SDPO losses are
combined using their global routed-token counts, without a manually tuned
mixing coefficient.

The default paper configuration uses `K=8`, `beta=1.0`, five ROPSD epochs,
and a 4,096-token response budget. DGPR uses `M=8` inference rollouts and a
cell-wise consensus threshold of `0.75`.

## Paper terminology and code

| Paper component | Main implementation |
| --- | --- |
| CRFT | [`prepare_reasoning_sft.py`](cell_annotation/prepare_reasoning_sft.py), [`sft_lora.py`](cell_annotation/sft_lora.py) |
| Dense reward and routing metadata | [`reward_cello1.py`](cell_annotation/reward_cello1.py) |
| ROPSD configuration | [`scopd_ropsd.yaml`](verl/trainer/config/scopd_ropsd.yaml) |
| Routed GRPO/SDPO objective | [`dp_actor.py`](verl/workers/actor/dp_actor.py), [`core_algos.py`](verl/trainer/ppo/core_algos.py) |
| DGPR | [`dgpr.py`](cell_annotation/dgpr.py), [`infer_dgpr_vllm.py`](cell_annotation/infer_dgpr_vllm.py) |

Historical experiment filenames are retained for checkpoint and ablation
reproducibility. The paper-facing launchers below are the recommended entry
points.

## Repository layout

| Path | Contents |
| --- | --- |
| `cell_annotation/` | Data preparation, CRFT, rewards, evaluation, DGPR, and experiment launchers |
| `cell_annotation/data/` | CellPuzzles source and processed CRFT/ROPSD/evaluation data |
| `cell_annotation/tests/` | Reward, routing, tokenization, evaluation, and DGPR unit tests |
| `verl/` | VERL trainer extended with routed GRPO/SDPO, EMA self-teaching, O1 fallback, and entropy-aware weighting |
| `verl/trainer/config/` | Main method and ablation configurations |
| `docs/EXPERIMENT_NOTES_ZH.md` | Detailed development and experiment notes in Chinese |
| `docs/UPSTREAM_SDPO_README.md` | Documentation of the upstream SDPO codebase |

## Data

The included data are derived from the public
[`ncbi/CellPuzzles`](https://huggingface.co/datasets/ncbi/CellPuzzles)
dataset and the evaluation-only unseen splits used by Cell-o1.

| Split | Rows |
| --- | ---: |
| CellPuzzles train | 6,912 |
| CellPuzzles test | 1,095 |
| Test-clean | 604 |
| Four unseen disease sets combined | 539 |
| O1-distilled reasoning trajectories | 3,912 |
| CRFT train/dev | 3,521 / 391 |

Manifests under `cell_annotation/data/**/manifest.json` record source URLs,
checksums, split seeds, prompt modes, and row counts. Large reproducible
intermediate arrays for baseline methods are not included. See
[`cell_annotation/data/README.md`](cell_annotation/data/README.md) for the
complete inventory and data-license notes.

To rebuild the canonical data:

```bash
export PYTHONPATH=$PWD
bash cell_annotation/scripts/download_and_prepare.sh
bash cell_annotation/scripts/prepare_o1_reasoning_sft_data.sh
python -m cell_annotation.prepare_nonthinking_reasoning_rl_data --help
python -m cell_annotation.prepare_o1_privileged_rl_data --help
```

## Installation

The training stack is based on
[`lasgroup/SDPO`](https://github.com/lasgroup/SDPO) and VERL. A CUDA machine is
required for model training and vLLM evaluation.

```bash
python -m pip install -e .
python -m pip install -r requirements/test.txt
```

Additional environment and container options are documented in
[`docs/INSTALL.md`](docs/INSTALL.md).

The experiments use Qwen3-8B and multi-GPU Ray jobs. Model weights and
training checkpoints are not stored in this repository.

## Reproduction

The launchers contain PJLab `rjob` defaults, but paths and hyperparameters can
be overridden through environment variables. Set `DRY_RUN=1` to inspect a job
without submitting it.

### Stage 1: CRFT

```bash
REPO_DIR=$PWD \
BASE_MODEL=/path/to/Qwen3-8B \
DRY_RUN=1 \
bash cell_annotation/cluster/submit_qwen3_8b_crft_rjob.sh
```

The canonical setup uses 3,912 O1-distilled trajectories, a 90/10 split, 10
epochs, LoRA rank 256 (`alpha=512`), and a maximum sequence length of 8,192.
Qwen3 runs with `enable_thinking=False`; only assistant completion tokens are
supervised.

### Stage 2: ROPSD

```bash
REPO_DIR=$PWD \
STUDENT_MODEL=/path/to/crft/merged_model \
DRY_RUN=1 \
bash cell_annotation/cluster/submit_qwen3_8b_ropsd_rjob.sh
```

The main entry point uses
[`compute_scopd_ropsd_score`](cell_annotation/reward_cello1.py) and
[`scopd_ropsd.yaml`](verl/trainer/config/scopd_ropsd.yaml). Adjacent launchers
provide the GRPO-only, sparse-reward, no-CRFT, no-O1, no-dynamic-weighting,
entropy-coefficient, and joint-objective ablations.

### Stage 3: DGPR

```bash
REPO_DIR=$PWD \
MODEL_PATH=/path/to/ropsd/merged_model \
DRY_RUN=1 \
bash cell_annotation/cluster/submit_qwen3_8b_dgpr_eval_rjob.sh
```

DGPR filters malformed, non-permutation, and reasoning-degenerate outputs;
selects an observed complete assignment as the Hamming medoid; localizes
stable and uncertain cells from cell-wise support; and accepts a refinement
only when it preserves stable assignments, uses rollout-supported labels, and
satisfies the global one-to-one constraint.

## Tests

Run the method unit tests from the repository root:

```bash
export PYTHONPATH=$PWD
python -m pytest -q \
  cell_annotation/tests/test_reward_cello1.py \
  cell_annotation/tests/test_sibling_srpo_core.py \
  cell_annotation/tests/test_dgpr.py \
  tests/trainer/ppo/test_metric_utils_on_cpu.py
```

GPU integration tests require the VERL/vLLM/Ray environment used for
training.

## Attribution and license

This repository modifies the Apache-2.0-licensed SDPO/VERL codebase. See
[`NOTICE`](NOTICE) and the upstream documentation for attribution.
CellPuzzles and other third-party data remain subject to their original terms;
the repository's [Apache License 2.0](LICENSE) applies to code, not third-party
datasets.
