#!/usr/bin/env python3
"""Optional answer-format LoRA SFT before GT-privileged SDPO."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from cell_annotation.common import read_records
except ImportError:
    from common import read_records


def _as_token_id_list(value: Any, *, source: str) -> List[int]:
    """Normalize chat-template outputs across Transformers versions.

    Transformers 5.x may return a ``BatchEncoding`` from
    ``apply_chat_template(tokenize=True)``.  Slicing that object yields
    ``tokenizers.Encoding`` objects instead of token ids, which only fails
    later in the data collator.  Older Transformers versions return a plain
    list.  Accept both representations and always return one unbatched list
    of integer token ids.
    """

    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise TypeError(f"{source} mapping has no input_ids")
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()

    # A tokenizers.Encoding exposes its integer ids via ``.ids``.  A
    # single-example BatchEncoding can also expose a one-element batch.
    if hasattr(value, "ids"):
        value = value.ids
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1:
        first = value[0]
        if isinstance(first, torch.Tensor):
            first = first.detach().cpu().tolist()
        if hasattr(first, "ids"):
            first = first.ids
        if isinstance(first, (list, tuple)):
            value = list(first)

    if not isinstance(value, list):
        raise TypeError(
            f"{source} returned unsupported token container {type(value).__name__}"
        )
    if not all(isinstance(token_id, Integral) for token_id in value):
        element_types = sorted({type(item).__name__ for item in value})
        raise TypeError(
            f"{source} contains non-integer token values: {element_types}"
        )
    return [int(token_id) for token_id in value]


class CellSFTDataset(Dataset):
    def __init__(
        self, path: Path, tokenizer: Any, max_length: int, target_mode: str
    ):
        self.rows = read_records(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_mode = target_mode

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(
        self, index: int
    ) -> tuple[Dict[str, List[int]], Dict[str, int]]:
        row = self.rows[index]
        messages = [
            {"role": "system", "content": row["system_msg"]},
            {"role": "user", "content": row["user_msg"]},
        ]
        if self.target_mode in {
            "reasoning",
            "nonthinking_reasoning",
            "explicit_reasoning",
        }:
            assistant = row.get("assistant_msg", "").strip()
            if not assistant:
                raise ValueError(
                    f"row {index} has no assistant_msg for reasoning SFT"
                )
            if self.target_mode in {
                "nonthinking_reasoning",
                "explicit_reasoning",
            } and not (
                assistant.startswith("<reasoning>")
                and "</reasoning>" in assistant
                and "<answer>" in assistant
                and assistant.endswith("</answer>")
                and "<think>" not in assistant
                and "</think>" not in assistant
            ):
                raise ValueError(
                    f"row {index} has invalid explicit reasoning target"
                )
        else:
            # Qwen3's non-thinking chat template pre-fills an empty think block.
            # Train only the answer continuation to match non-thinking rollout.
            assistant = f"<answer>{row['answer']}</answer>"
        chat_template_kwargs: Dict[str, Any] = {}
        if self.target_mode == "reasoning":
            chat_template_kwargs["enable_thinking"] = True
        elif self.target_mode == "nonthinking_reasoning":
            chat_template_kwargs["enable_thinking"] = False
        prompt_ids = _as_token_id_list(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **chat_template_kwargs,
            ),
            source="generation chat template",
        )
        if self.target_mode in {"nonthinking_reasoning", "explicit_reasoning"}:
            # Some Qwen chat templates include an empty <think></think> block
            # in the non-thinking generation prompt. Re-rendering a complete assistant
            # message and recovering its target with a common-prefix diff is
            # not stable across Qwen chat-template versions: in Transformers
            # 5.x the rendered complete message can be a prefix of the
            # generation prompt, yielding an empty supervised target.
            #
            # Build exactly the sequence used at inference instead: retain the
            # non-thinking generation prompt and explicitly append the visible
            # <reasoning>...<answer> completion plus the chat EOS token.
            assistant_ids = _as_token_id_list(
                self.tokenizer.encode(assistant, add_special_tokens=False),
                source="assistant completion encoding",
            )
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer has no EOS token for SFT target")
            target_ids = assistant_ids + [self.tokenizer.eos_token_id]
        else:
            full_ids = _as_token_id_list(
                self.tokenizer.apply_chat_template(
                    messages + [{"role": "assistant", "content": assistant}],
                    tokenize=True,
                    add_generation_prompt=False,
                    **chat_template_kwargs,
                ),
                source="complete chat template",
            )
            prefix = 0
            for prompt_token, full_token in zip(prompt_ids, full_ids):
                if prompt_token != full_token:
                    break
                prefix += 1
            target_ids = full_ids[prefix:]
        if not target_ids:
            raise ValueError("chat template produced an empty SFT target")
        raw_prompt_length = len(prompt_ids)
        raw_target_length = len(target_ids)
        if len(target_ids) >= self.max_length:
            target_ids = target_ids[-(self.max_length - 1) :]
        prompt_budget = self.max_length - len(target_ids)
        kept_prompt = prompt_ids[-prompt_budget:]
        input_ids = kept_prompt + target_ids
        labels = [-100] * len(kept_prompt) + target_ids
        encoded = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }
        metadata = {
            "raw_prompt_length": raw_prompt_length,
            "raw_target_length": raw_target_length,
            "raw_total_length": raw_prompt_length + raw_target_length,
            "kept_prompt_length": len(kept_prompt),
            "kept_target_length": len(target_ids),
            "kept_total_length": len(input_ids),
            "prompt_truncated": int(len(kept_prompt) < raw_prompt_length),
            "target_truncated": int(len(target_ids) < raw_target_length),
        }
        return encoded, metadata

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        encoded, _ = self._encode(index)
        return encoded

    def audit_lengths(self) -> Dict[str, Any]:
        metadata = [self._encode(index)[1] for index in range(len(self))]
        raw_total_lengths = sorted(item["raw_total_length"] for item in metadata)
        raw_target_lengths = sorted(item["raw_target_length"] for item in metadata)

        def percentile(values: List[int], fraction: float) -> int:
            if not values:
                return 0
            index = min(len(values) - 1, round((len(values) - 1) * fraction))
            return values[index]

        return {
            "examples": len(metadata),
            "max_length": self.max_length,
            "raw_total_tokens": {
                "min": min(raw_total_lengths),
                "median": round(statistics.median(raw_total_lengths)),
                "p95": percentile(raw_total_lengths, 0.95),
                "p99": percentile(raw_total_lengths, 0.99),
                "max": max(raw_total_lengths),
            },
            "raw_target_tokens": {
                "min": min(raw_target_lengths),
                "median": round(statistics.median(raw_target_lengths)),
                "p95": percentile(raw_target_lengths, 0.95),
                "p99": percentile(raw_target_lengths, 0.99),
                "max": max(raw_target_lengths),
            },
            "prompt_truncated_examples": sum(
                item["prompt_truncated"] for item in metadata
            ),
            "target_truncated_examples": sum(
                item["target_truncated"] for item in metadata
            ),
        }


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch: Dict[str, List[List[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_token_id] * padding
            )
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-mode",
        choices=(
            "answer",
            "reasoning",
            "nonthinking_reasoning",
            "explicit_reasoning",
        ),
        default="answer",
        help=(
            "answer trains only <answer>; reasoning trains <think> with "
            "Qwen3 enable_thinking=True; nonthinking_reasoning trains an "
            "explicit <reasoning> response with enable_thinking=False; "
            "explicit_reasoning uses <reasoning> with no thinking-template "
            "argument for models such as Llama 3.1"
        ),
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="flash_attention_2",
        help=(
            "Attention backend used while loading the model. Use sdpa if the "
            "model's FlashAttention path is unavailable or incompatible."
        ),
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--use-dora",
        action="store_true",
        help="Use weight-decomposed LoRA, as in the original Cell-o1 SFT recipe.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument(
        "--eval-strategy", choices=("no", "steps", "epoch"), default="steps"
    )
    parser.add_argument(
        "--save-strategy", choices=("no", "steps", "epoch"), default="steps"
    )
    parser.add_argument(
        "--save-only-model",
        action="store_true",
        help=(
            "Save model/adapter weights without optimizer state. This makes "
            "checkpoints much smaller, but they cannot be used to resume training."
        ),
    )
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume from a complete Trainer checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fail-on-target-truncation",
        action="store_true",
        help="Abort before training if any supervised assistant target is truncated.",
    )
    parser.add_argument("--report-to-wandb", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_dora=args.use_dora,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    if int(os.getenv("RANK", "0")) == 0:
        model.print_trainable_parameters()

    train_dataset = CellSFTDataset(
        args.train_file, tokenizer, args.max_length, args.target_mode
    )
    dev_dataset = CellSFTDataset(
        args.dev_file, tokenizer, args.max_length, args.target_mode
    )
    if int(os.getenv("RANK", "0")) == 0:
        length_audit = {
            "train": train_dataset.audit_lengths(),
            "dev": dev_dataset.audit_lengths(),
        }
        print("[length-audit]")
        print(json.dumps(length_audit, indent=2, ensure_ascii=False))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "length_audit.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(length_audit, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        target_truncation_count = sum(
            split["target_truncated_examples"] for split in length_audit.values()
        )
        if args.fail_on_target_truncation and target_truncation_count:
            raise RuntimeError(
                "supervised reasoning targets would be truncated: "
                f"{target_truncation_count} examples"
            )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=0.0,
        report_to=["wandb"] if args.report_to_wandb else [],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
    )
    resume_from_checkpoint = None
    if args.resume_from_checkpoint is not None:
        if not args.resume_from_checkpoint.is_dir():
            raise FileNotFoundError(
                f"resume checkpoint does not exist: {args.resume_from_checkpoint}"
            )
        resume_from_checkpoint = str(args.resume_from_checkpoint)
        if int(os.getenv("RANK", "0")) == 0:
            print(f"[resume] {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        print(f"[done] LoRA adapter: {final_dir}")


if __name__ == "__main__":
    main()
