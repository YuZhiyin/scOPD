#!/usr/bin/env python3
"""M=8 disagreement-gated pairwise refinement with a single vLLM engine."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from cell_annotation.common import write_json, write_jsonl
    from cell_annotation.dgpr import (
        ParsedRollout,
        build_dgpr_plan,
        build_refinement_instruction,
        compact_trace,
        parse_candidate_labels,
        parse_rollout,
        plan_to_dict,
        validate_refinement,
    )
    from cell_annotation.infer_vllm import system_prompt_for_mode
except ImportError:
    from common import write_json, write_jsonl  # type: ignore
    from dgpr import (  # type: ignore
        ParsedRollout,
        build_dgpr_plan,
        build_refinement_instruction,
        compact_trace,
        parse_candidate_labels,
        parse_rollout,
        plan_to_dict,
        validate_refinement,
    )
    from infer_vllm import system_prompt_for_mode  # type: ignore


SPLIT_FILES = OrderedDict(
    [
        ("test", "test.json"),
        ("test_clean", "test_clean.json"),
        ("unseen_all", "unseen_all.json"),
        ("breast_cancer", "unseen/breast_cancer.json"),
        ("colorectal_cancer", "unseen/colorectal_cancer.json"),
        ("melanoma", "unseen/melanoma.json"),
        ("systemic_lupus_erythematosus", "unseen/systemic_lupus_erythematosus.json"),
    ]
)


def _load_inputs(data_dir: Path, limit: int | None) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    split_ids: Dict[str, List[str]] = {}
    unique: Dict[str, Dict[str, Any]] = OrderedDict()
    for split_name, relative_path in SPLIT_FILES.items():
        path = data_dir / relative_path
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if limit is not None:
            rows = rows[:limit]
        ids = []
        for row_index, row in enumerate(rows):
            row_id = str(row.get("id", f"{split_name}_{row_index}"))
            ids.append(row_id)
            if row_id in unique:
                previous = unique[row_id]
                for key in ("system_msg", "user_msg", "answer"):
                    if previous.get(key) != row.get(key):
                        raise ValueError(f"duplicate id {row_id!r} disagrees on {key}")
            else:
                unique[row_id] = dict(row)
        split_ids[split_name] = ids
    return split_ids, unique


def _chat_prompt(tokenizer: Any, row: Dict[str, Any], user_content: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": system_prompt_for_mode(
                    row["system_msg"], enable_thinking=False, nonthinking_reasoning=True
                ),
            },
            {"role": "user", "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _fit_refinement_prompt(
    tokenizer: Any,
    row: Dict[str, Any],
    anchor: ParsedRollout,
    challenger: ParsedRollout,
    plan: Any,
    candidates: Sequence[str],
    max_input_tokens: int,
) -> Tuple[str | None, int | None, int | None]:
    for max_chars in (100_000, 24_000, 16_000, 12_000, 8_000, 4_000):
        compact_anchor = ParsedRollout(
            **{**anchor.__dict__, "text": compact_trace(anchor.text, max_chars)}
        )
        compact_challenger = ParsedRollout(
            **{**challenger.__dict__, "text": compact_trace(challenger.text, max_chars)}
        )
        instruction = build_refinement_instruction(
            compact_anchor, compact_challenger, plan, candidates
        )
        prompt = _chat_prompt(tokenizer, row, row["user_msg"] + instruction)
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        if token_count <= max_input_tokens:
            return prompt, token_count, max_chars
    return None, None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--rollout-temperature", type=float, default=0.6)
    parser.add_argument("--rollout-top-p", type=float, default=0.95)
    parser.add_argument("--refinement-temperature", type=float, default=0.2)
    parser.add_argument("--refinement-top-p", type=float, default=0.9)
    parser.add_argument("--consensus-threshold", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.num_rollouts < 2:
        parser.error("--num-rollouts must be at least 2")
    max_refinement_input_tokens = args.max_model_len - args.max_new_tokens - 256
    if max_refinement_input_tokens <= 0:
        parser.error("max-model-len must exceed max-new-tokens by at least 256")

    split_ids, unique_rows = _load_inputs(args.data_dir, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    row_ids = list(unique_rows)
    rows = [unique_rows[row_id] for row_id in row_ids]
    initial_prompts = [_chat_prompt(tokenizer, row, row["user_msg"]) for row in rows]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    rollout_sampling = SamplingParams(
        temperature=args.rollout_temperature,
        top_p=args.rollout_top_p,
        max_tokens=args.max_new_tokens,
        n=args.num_rollouts,
        seed=args.seed,
    )
    initial_outputs = llm.generate(initial_prompts, rollout_sampling, use_tqdm=True)

    states: Dict[str, Dict[str, Any]] = OrderedDict()
    refinement_prompts: List[str] = []
    refinement_ids: List[str] = []
    for row_id, row, request_output in zip(row_ids, rows, initial_outputs):
        candidates = parse_candidate_labels(row["user_msg"])
        parsed_rollouts = [
            parse_rollout(index, candidate.text, candidates)
            for index, candidate in enumerate(request_output.outputs)
        ]
        plan = build_dgpr_plan(
            parsed_rollouts,
            num_cells=len(candidates),
            consensus_threshold=args.consensus_threshold,
        )
        if plan.anchor_index is None:
            # Preserve a diagnostic output even when no strict rollout exists.
            anchor_text = request_output.outputs[0].text
            anchor = parsed_rollouts[0]
        else:
            anchor = parsed_rollouts[plan.anchor_index]
            anchor_text = anchor.text
        state: Dict[str, Any] = {
            "row": row,
            "candidates": candidates,
            "rollouts": parsed_rollouts,
            "finish_reasons": [output.finish_reason for output in request_output.outputs],
            "plan": plan,
            "anchor": anchor,
            "anchor_prediction": anchor_text,
            "final_prediction": anchor_text,
            "refinement_prediction": None,
            "refinement_accepted": False,
            "refinement_status": "not_triggered",
            "refinement_prompt_tokens": None,
            "attempt_max_reasoning_chars": None,
        }
        if plan.trigger_refinement and plan.challenger_index is not None:
            challenger = parsed_rollouts[plan.challenger_index]
            prompt, prompt_tokens, max_chars = _fit_refinement_prompt(
                tokenizer,
                row,
                anchor,
                challenger,
                plan,
                candidates,
                max_refinement_input_tokens,
            )
            if prompt is None:
                state["refinement_status"] = "prompt_too_long"
            else:
                refinement_ids.append(row_id)
                refinement_prompts.append(prompt)
                state["refinement_status"] = "pending"
                state["refinement_prompt_tokens"] = prompt_tokens
                state["attempt_max_reasoning_chars"] = max_chars
        states[row_id] = state

    if refinement_prompts:
        refinement_sampling = SamplingParams(
            temperature=args.refinement_temperature,
            top_p=args.refinement_top_p,
            max_tokens=args.max_new_tokens,
            n=1,
            seed=args.seed + 100_000,
        )
        refinement_outputs = llm.generate(
            refinement_prompts, refinement_sampling, use_tqdm=True
        )
        for row_id, output in zip(refinement_ids, refinement_outputs):
            state = states[row_id]
            text = output.outputs[0].text
            accepted, reason, _ = validate_refinement(
                text,
                state["candidates"],
                state["anchor"],
                state["plan"],
            )
            state["refinement_prediction"] = text
            state["refinement_finish_reason"] = output.outputs[0].finish_reason
            state["refinement_accepted"] = accepted
            state["refinement_status"] = reason
            if accepted:
                state["final_prediction"] = text

    raw_rows = []
    for row_id, state in states.items():
        raw_rows.append(
            {
                "id": row_id,
                "rollouts": [
                    {
                        "index": rollout.index,
                        "prediction": rollout.text,
                        "valid": rollout.valid,
                        "invalid_reason": rollout.invalid_reason,
                        "assignments": list(rollout.assignments),
                        "reasoning_chars": rollout.reasoning_chars,
                        "reasoning_units": rollout.reasoning_units,
                        "finish_reason": state["finish_reasons"][rollout.index],
                    }
                    for rollout in state["rollouts"]
                ],
                "plan": plan_to_dict(state["plan"]),
                "refinement_prediction": state["refinement_prediction"],
                "refinement_accepted": state["refinement_accepted"],
                "refinement_status": state["refinement_status"],
                "refinement_prompt_tokens": state["refinement_prompt_tokens"],
                "attempt_max_reasoning_chars": state["attempt_max_reasoning_chars"],
            }
        )
    write_jsonl(args.output_dir / "raw" / "unique_dgpr.jsonl", raw_rows)

    for split_name, ids in split_ids.items():
        anchor_rows = []
        final_rows = []
        for row_id in ids:
            state = states[row_id]
            diagnostic = {
                **plan_to_dict(state["plan"]),
                "refinement_accepted": state["refinement_accepted"],
                "refinement_status": state["refinement_status"],
                "refinement_prompt_tokens": state["refinement_prompt_tokens"],
            }
            anchor_row = dict(state["row"])
            anchor_row["prediction"] = state["anchor_prediction"]
            anchor_row["dgpr"] = diagnostic
            anchor_rows.append(anchor_row)
            final_row = dict(state["row"])
            final_row["prediction"] = state["final_prediction"]
            final_row["dgpr"] = diagnostic
            final_rows.append(final_row)
        write_json(
            args.output_dir / "predictions" / f"{split_name}_anchor_medoid.json",
            anchor_rows,
        )
        write_json(args.output_dir / "predictions" / f"{split_name}.json", final_rows)

    manifest = {
        "model": args.model,
        "num_unique_inputs": len(rows),
        "split_sizes": {key: len(value) for key, value in split_ids.items()},
        "num_rollouts": args.num_rollouts,
        "rollout_temperature": args.rollout_temperature,
        "rollout_top_p": args.rollout_top_p,
        "refinement_temperature": args.refinement_temperature,
        "refinement_top_p": args.refinement_top_p,
        "consensus_threshold": args.consensus_threshold,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "selection_uses_ground_truth": False,
    }
    write_json(args.output_dir / "dgpr_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

