# Data inventory

This directory contains the data needed to reproduce the Qwen3 scOPD
experiments. Data provenance and deterministic split details are recorded in
the JSON manifests next to each dataset.

## Included

| Directory | Purpose |
| --- | --- |
| `raw/` | Official CellPuzzles train, test, and reasoning parquet files |
| `processed/` | Canonical train/test/test-clean/unseen data and SFT files |
| `cello1_nonthinking_reasoning_sft_reproduction/` | Main non-thinking explicit-reasoning SFT split |
| `cello1_reasoning_sft_reproduction/` | Thinking-mode SFT reproduction split |
| `cello1_llama_explicit_reasoning_sft_reproduction/` | Llama explicit-reasoning transfer split |
| `rl_nonthinking_reasoning_o1_privileged/` | Main scOPD train/test parquet with offline-o1 availability |
| `rl_explicit_reasoning_o1_privileged/` | Explicit-reasoning transfer variant |
| `rl_thinking_o1_privileged/` | Thinking-mode privileged-feedback variant |
| `rl_thinking/` | Cell-o1-aligned thinking-mode data and evaluation splits |

## Deliberately omitted generated caches

The following are not inputs to scOPD and can be recreated with the included
scripts:

* `genept_s_ada002/`: OpenAI embedding cache and chunk files;
* `processed/c2s/`: expanded C2S fine-tuning cache;
* `rl_thinking/raw/`: byte-identical duplicates of `raw/train` and `raw/test`.

Model checkpoints, optimizer states, rollouts, predictions, and logs are also
excluded from Git.

## Provenance and terms

The official train/test/reasoning files were downloaded from
[`ncbi/CellPuzzles`](https://huggingface.co/datasets/ncbi/CellPuzzles). The
four unseen evaluation splits are derived from the evaluation-only Cell-o1
data. The code license in the repository root does not relicense these
third-party datasets; users are responsible for following the source dataset
terms and citation requirements.
