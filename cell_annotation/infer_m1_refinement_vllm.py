#!/usr/bin/env python3
"""One greedy rollout, one short critic, and at most one refinement."""

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
    from cell_annotation.dgpr import CANDIDATE_HEADER, compact_trace, parse_candidate_labels, parse_rollout
    from cell_annotation.infer_vllm import system_prompt_for_mode
    from cell_annotation.m1_refinement import (
        build_critic_instruction,
        build_refinement_instruction,
        decision_to_dict,
        parse_cell_genes,
        validate_refinement,
        verify_critic_output,
    )
except ImportError:
    from common import write_json, write_jsonl  # type: ignore
    from dgpr import CANDIDATE_HEADER, compact_trace, parse_candidate_labels, parse_rollout  # type: ignore
    from infer_vllm import system_prompt_for_mode  # type: ignore
    from m1_refinement import (  # type: ignore
        build_critic_instruction,
        build_refinement_instruction,
        decision_to_dict,
        parse_cell_genes,
        validate_refinement,
        verify_critic_output,
    )


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
CRITIC_SYSTEM = (
    "You are a conservative verifier for single-cell annotation. Inspect the "
    "given genes and annotation attempt, then return only the requested JSON. "
    "Never invent genes and never use hidden or ground-truth labels."
)


def _load_inputs(data_dir: Path, limit: int | None) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    split_ids: Dict[str, List[str]] = {}
    unique: Dict[str, Dict[str, Any]] = OrderedDict()
    for split_name, relative_path in SPLIT_FILES.items():
        with (data_dir / relative_path).open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if limit is not None:
            rows = rows[:limit]
        ids: List[str] = []
        for row_index, row in enumerate(rows):
            row_id = str(row.get("id", f"{split_name}_{row_index}"))
            ids.append(row_id)
            if row_id in unique:
                previous = unique[row_id]
                for key in ("system_msg", "user_msg"):
                    if previous.get(key) != row.get(key):
                        raise ValueError(f"duplicate id {row_id!r} disagrees on {key}")
            else:
                unique[row_id] = dict(row)
        split_ids[split_name] = ids
    return split_ids, unique


def _generation_prompt(tokenizer: Any, row: Dict[str, Any], user_content: str) -> str:
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


def _critic_prompt(tokenizer: Any, row: Dict[str, Any], initial_text: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": CRITIC_SYSTEM},
            {
                "role": "user",
                "content": row["user_msg"] + build_critic_instruction(initial_text),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _fit_prompt(
    tokenizer: Any,
    row: Dict[str, Any],
    initial_text: str,
    instruction_builder: Any,
    max_input_tokens: int,
) -> Tuple[str | None, int | None, int | None]:
    for max_chars in (100_000, 24_000, 16_000, 12_000, 8_000, 4_000):
        compact = compact_trace(initial_text, max_chars)
        prompt = instruction_builder(compact)
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
    parser.add_argument("--critic-max-tokens", type=int, default=1024)
    parser.add_argument("--refinement-temperature", type=float, default=0.2)
    parser.add_argument("--refinement-top-p", type=float, default=0.9)
    parser.add_argument("--min-supporting-genes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    max_long_input = args.max_model_len - args.max_new_tokens - 256
    max_critic_input = args.max_model_len - args.critic_max_tokens - 256
    if min(max_long_input, max_critic_input) <= 0:
        parser.error("model context is too short for the requested output budgets")

    split_ids, unique_rows = _load_inputs(args.data_dir, args.limit)
    row_ids = list(unique_rows)
    rows = [unique_rows[row_id] for row_id in row_ids]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )

    initial_prompts = [_generation_prompt(tokenizer, row, row["user_msg"]) for row in rows]
    initial_outputs = llm.generate(
        initial_prompts,
        SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, n=1, seed=args.seed),
        use_tqdm=True,
    )

    states: Dict[str, Dict[str, Any]] = OrderedDict()
    critic_ids: List[str] = []
    critic_prompts: List[str] = []
    for row_id, row, request_output in zip(row_ids, rows, initial_outputs):
        initial_text = request_output.outputs[0].text
        candidates = parse_candidate_labels(row["user_msg"])
        cell_genes = parse_cell_genes(row["user_msg"], CANDIDATE_HEADER)
        if len(cell_genes) != len(candidates):
            raise ValueError(f"{row_id}: cell count and candidate count differ")
        initial = parse_rollout(0, initial_text, candidates)
        state: Dict[str, Any] = {
            "row": row,
            "candidates": candidates,
            "cell_genes": cell_genes,
            "initial": initial,
            "initial_prediction": initial_text,
            "initial_finish_reason": request_output.outputs[0].finish_reason,
            "critic_prediction": None,
            "critic_decision": None,
            "refinement_prediction": None,
            "refinement_accepted": False,
            "refinement_status": "pending_structural_repair" if not initial.valid else "pending_critic",
            "final_prediction": initial_text,
            "critic_prompt_tokens": None,
            "refinement_prompt_tokens": None,
        }
        if initial.valid:
            prompt, tokens, _ = _fit_prompt(
                tokenizer,
                row,
                initial_text,
                lambda compact: _critic_prompt(tokenizer, row, compact),
                max_critic_input,
            )
            if prompt is None:
                state["refinement_status"] = "critic_prompt_too_long"
            else:
                critic_ids.append(row_id)
                critic_prompts.append(prompt)
                state["critic_prompt_tokens"] = tokens
        states[row_id] = state

    if critic_prompts:
        critic_outputs = llm.generate(
            critic_prompts,
            SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.critic_max_tokens, n=1, seed=args.seed + 10_000),
            use_tqdm=True,
        )
        for row_id, output in zip(critic_ids, critic_outputs):
            state = states[row_id]
            critic_text = output.outputs[0].text
            decision = verify_critic_output(
                critic_text,
                state["initial"],
                state["candidates"],
                state["cell_genes"],
                min_supporting_genes=args.min_supporting_genes,
            )
            state["critic_prediction"] = critic_text
            state["critic_finish_reason"] = output.outputs[0].finish_reason
            state["critic_decision"] = decision
            state["refinement_status"] = decision.status

    refinement_ids: List[str] = []
    refinement_prompts: List[str] = []
    for row_id, state in states.items():
        initial = state["initial"]
        decision = state["critic_decision"]
        issue = decision.issue if decision is not None and decision.valid else None
        should_refine = not initial.valid or issue is not None
        if not should_refine:
            continue
        structural_reason = initial.invalid_reason if not initial.valid else None
        prompt, tokens, _ = _fit_prompt(
            tokenizer,
            state["row"],
            state["initial_prediction"],
            lambda compact: _generation_prompt(
                tokenizer,
                state["row"],
                state["row"]["user_msg"]
                + build_refinement_instruction(compact, issue, structural_reason),
            ),
            max_long_input,
        )
        if prompt is None:
            state["refinement_status"] = "refinement_prompt_too_long"
            continue
        refinement_ids.append(row_id)
        refinement_prompts.append(prompt)
        state["refinement_prompt_tokens"] = tokens
        state["refinement_status"] = "pending_refinement"

    if refinement_prompts:
        refinement_outputs = llm.generate(
            refinement_prompts,
            SamplingParams(
                temperature=args.refinement_temperature,
                top_p=args.refinement_top_p,
                max_tokens=args.max_new_tokens,
                n=1,
                seed=args.seed + 20_000,
            ),
            use_tqdm=True,
        )
        for row_id, output in zip(refinement_ids, refinement_outputs):
            state = states[row_id]
            text = output.outputs[0].text
            decision = state["critic_decision"]
            issue = decision.issue if decision is not None and decision.valid else None
            accepted, reason, _ = validate_refinement(
                text, state["candidates"], state["initial"], issue
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
                "initial_prediction": state["initial_prediction"],
                "initial_valid": state["initial"].valid,
                "initial_invalid_reason": state["initial"].invalid_reason,
                "initial_finish_reason": state["initial_finish_reason"],
                "critic_prediction": state["critic_prediction"],
                "critic": decision_to_dict(state["critic_decision"]),
                "refinement_prediction": state["refinement_prediction"],
                "refinement_accepted": state["refinement_accepted"],
                "refinement_status": state["refinement_status"],
                "critic_prompt_tokens": state["critic_prompt_tokens"],
                "refinement_prompt_tokens": state["refinement_prompt_tokens"],
            }
        )
    write_jsonl(args.output_dir / "raw" / "unique_m1_refinement.jsonl", raw_rows)

    for split_name, ids in split_ids.items():
        initial_rows = []
        final_rows = []
        for row_id in ids:
            state = states[row_id]
            diagnostic = {
                "initial_valid": state["initial"].valid,
                "initial_invalid_reason": state["initial"].invalid_reason,
                "critic": decision_to_dict(state["critic_decision"]),
                "trigger_refinement": bool(
                    not state["initial"].valid
                    or (
                        state["critic_decision"] is not None
                        and state["critic_decision"].valid
                        and state["critic_decision"].issue is not None
                    )
                ),
                "refinement_accepted": state["refinement_accepted"],
                "refinement_status": state["refinement_status"],
            }
            initial_row = dict(state["row"])
            initial_row["prediction"] = state["initial_prediction"]
            initial_row["m1_refinement"] = diagnostic
            initial_rows.append(initial_row)
            final_row = dict(state["row"])
            final_row["prediction"] = state["final_prediction"]
            final_row["m1_refinement"] = diagnostic
            final_rows.append(final_row)
        write_json(args.output_dir / "predictions" / f"{split_name}_initial.json", initial_rows)
        write_json(args.output_dir / "predictions" / f"{split_name}.json", final_rows)

    manifest = {
        "model": args.model,
        "num_unique_inputs": len(rows),
        "split_sizes": {name: len(ids) for name, ids in split_ids.items()},
        "initial_temperature": 0.0,
        "critic_temperature": 0.0,
        "critic_max_tokens": args.critic_max_tokens,
        "refinement_temperature": args.refinement_temperature,
        "refinement_top_p": args.refinement_top_p,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "min_supporting_genes": args.min_supporting_genes,
        "maximum_refinements": 1,
        "selection_uses_ground_truth": False,
        "seed": args.seed,
    }
    write_json(args.output_dir / "m1_refinement_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

