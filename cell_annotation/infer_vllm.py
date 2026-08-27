#!/usr/bin/env python3
"""Batched Qwen3 inference with explicit thinking/non-thinking chat mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from cell_annotation.common import read_records, write_json
except ImportError:
    from common import read_records, write_json


ORIGINAL_REASONING_INSTRUCTION = (
    "Include your detailed reasoning within <think> and </think> tags, and "
    "provide your final answer within <answer> and </answer> tags."
)
NONTHINKING_INSTRUCTION = (
    "The model is evaluated in non-thinking mode, so do not provide reasoning. "
    "Provide only the final answer within <answer> and </answer> tags."
)
NONTHINKING_REASONING_INSTRUCTION = (
    "The model operates in non-thinking mode. Provide explicit, detailed "
    "biological reasoning within <reasoning> and </reasoning> tags. Ground "
    "each assignment in marker genes, cell and donor context, and the global "
    "one-to-one matching constraint. Then provide your final answer within "
    "<answer> and </answer> tags."
)


def system_prompt_for_mode(
    system_msg: str,
    enable_thinking: bool,
    nonthinking_reasoning: bool = False,
    native_cello1: bool = False,
) -> str:
    if sum(bool(value) for value in (enable_thinking, nonthinking_reasoning, native_cello1)) > 1:
        raise ValueError(
            "enable_thinking, nonthinking_reasoning, and native_cello1 are mutually exclusive"
        )
    if native_cello1:
        if NONTHINKING_INSTRUCTION in system_msg:
            return system_msg.replace(
                NONTHINKING_INSTRUCTION,
                ORIGINAL_REASONING_INSTRUCTION,
            )
        return system_msg
    if nonthinking_reasoning:
        if NONTHINKING_INSTRUCTION in system_msg:
            return system_msg.replace(
                NONTHINKING_INSTRUCTION,
                NONTHINKING_REASONING_INSTRUCTION,
            )
        if ORIGINAL_REASONING_INSTRUCTION in system_msg:
            return system_msg.replace(
                ORIGINAL_REASONING_INSTRUCTION,
                NONTHINKING_REASONING_INSTRUCTION,
            )
        return f"{system_msg.rstrip()} {NONTHINKING_REASONING_INSTRUCTION}"
    if not enable_thinking:
        return system_msg
    if NONTHINKING_INSTRUCTION in system_msg:
        return system_msg.replace(
            NONTHINKING_INSTRUCTION, ORIGINAL_REASONING_INSTRUCTION
        )
    return system_msg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Use Qwen3 enable_thinking=True and restore the original CellPuzzles "
            "reasoning instruction when processed data contains the non-thinking one"
        ),
    )
    parser.add_argument(
        "--nonthinking-reasoning",
        action="store_true",
        help=(
            "Use Qwen3 enable_thinking=False while requesting an explicit "
            "<reasoning>...</reasoning><answer>...</answer> response"
        ),
    )
    parser.add_argument(
        "--native-cello1",
        action="store_true",
        help=(
            "Restore Cell-o1's native <think> reasoning instruction without "
            "passing Qwen3's enable_thinking template argument"
        ),
    )
    args = parser.parse_args()
    if sum(
        bool(value)
        for value in (
            args.enable_thinking,
            args.nonthinking_reasoning,
            args.native_cello1,
        )
    ) > 1:
        parser.error(
            "--enable-thinking, --nonthinking-reasoning, and --native-cello1 "
            "cannot be combined"
        )

    records = read_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    template_kwargs = (
        {} if args.native_cello1 else {"enable_thinking": args.enable_thinking}
    )
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": system_prompt_for_mode(
                        row["system_msg"],
                        args.enable_thinking,
                        args.nonthinking_reasoning,
                        args.native_cello1,
                    ),
                },
                {"role": "user", "content": row["user_msg"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        for row in records
    ]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=1,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    result = []
    for row, output in zip(records, outputs):
        updated = dict(row)
        updated["prediction"] = output.outputs[0].text
        updated["finish_reason"] = output.outputs[0].finish_reason
        result.append(updated)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "rows": len(result),
                "enable_thinking": args.enable_thinking,
                "nonthinking_reasoning": args.nonthinking_reasoning,
                "native_cello1": args.native_cello1,
            }
        )
    )


if __name__ == "__main__":
    main()
