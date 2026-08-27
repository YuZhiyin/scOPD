#!/usr/bin/env python3
"""Full fine-tuning of a C2S causal LM on independent CellPuzzles cells."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from cell_annotation.common import read_records, write_json
except ImportError:
    from common import read_records, write_json


class JsonlDataset:
    """Minimal map-style dataset that keeps raw prompt/response strings."""

    def __init__(self, path: Path):
        self.rows = read_records(path)
        if not self.rows:
            raise ValueError(f"empty training dataset: {path}")
        for index, row in enumerate(self.rows):
            if not row.get("prompt") or not row.get("response"):
                raise ValueError(f"{path}: row {index} lacks prompt/response")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]


class CausalPromptResponseCollator:
    """Tokenize and right-pad C2S prompts without silently truncating them."""

    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
        response_only_loss: bool,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.response_only_loss = response_only_loss
        if tokenizer.eos_token_id is None:
            raise ValueError("C2S tokenizer must define eos_token_id")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def _encode(self, row: Dict[str, Any]) -> tuple[List[int], List[int]]:
        prompt_ids = self.tokenizer.encode(
            row["prompt"], add_special_tokens=False
        )
        response_ids = self.tokenizer.encode(
            row["response"], add_special_tokens=False
        )
        eos_id = int(self.tokenizer.eos_token_id)
        if not response_ids or response_ids[-1] != eos_id:
            response_ids.append(eos_id)
        input_ids = prompt_ids + response_ids
        if len(input_ids) > self.max_length:
            raise ValueError(
                f"{row.get('id')}: tokenized length {len(input_ids)} exceeds "
                f"--max-length {self.max_length}; refusing to truncate genes "
                "or candidate labels"
            )
        if self.response_only_loss:
            labels = [-100] * len(prompt_ids) + list(response_ids)
        else:
            labels = list(input_ids)
        return input_ids, labels

    def __call__(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        encoded = [self._encode(row) for row in rows]
        max_length = max(len(item[0]) for item in encoded)
        pad_id = int(self.tokenizer.pad_token_id)
        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []
        for input_ids, labels in encoded:
            pad_count = max_length - len(input_ids)
            batch_input_ids.append(input_ids + [pad_id] * pad_count)
            batch_attention_mask.append(
                [1] * len(input_ids) + [0] * pad_count
            )
            batch_labels.append(labels + [-100] * pad_count)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                batch_attention_mask, dtype=torch.long
            ),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--response-only-loss", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.per_device_batch_size <= 0:
        raise ValueError("--per-device-batch-size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("C2S fine-tuning requires a CUDA GPU")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset = JsonlDataset(args.train_file)
    dev_dataset = JsonlDataset(args.dev_file)
    collator = CausalPromptResponseCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        response_only_loss=args.response_only_loss,
    )

    training_kwargs: Dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_batch_size,
        "per_device_eval_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "cosine",
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "dataloader_num_workers": args.num_workers,
        "remove_unused_columns": False,
        "report_to": [],
        "seed": args.seed,
        "data_seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "ddp_find_unused_parameters": False,
        "save_safetensors": True,
    }
    training_parameters = inspect.signature(
        TrainingArguments.__init__
    ).parameters
    if "eval_strategy" in training_parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "data_collator": collator,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    train_result = trainer.train()

    final_model_dir = args.output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(final_model_dir)
    manifest = {
        "base_model": args.model,
        "train_file": str(args.train_file),
        "dev_file": str(args.dev_file),
        "train_examples": len(train_dataset),
        "dev_examples": len(dev_dataset),
        "final_model": str(final_model_dir),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "train_metrics": train_result.metrics,
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "world_size": int(
                __import__("os").environ.get("WORLD_SIZE", "1")
            ),
            "max_length": args.max_length,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "scheduler": "cosine",
            "response_only_loss": args.response_only_loss,
            "seed": args.seed,
        },
    }
    write_json(args.output_dir / "training_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
