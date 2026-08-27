#!/usr/bin/env python3
"""Independent closed-set cell-level inference for C2S causal LMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    from cell_annotation.c2s_common import (
        C2S_PROMPT_VERSION,
        build_c2s_prompt,
        c2s_target,
        select_independent,
    )
    from cell_annotation.common import read_records, write_json
    from cell_annotation.infer_chatcell import parse_cellpuzzles_prompt
except ImportError:
    from c2s_common import (
        C2S_PROMPT_VERSION,
        build_c2s_prompt,
        c2s_target,
        select_independent,
    )
    from common import read_records, write_json
    from infer_chatcell import parse_cellpuzzles_prompt


def resolve_dtype(torch_module: Any, name: str, device: str) -> Any:
    if name == "auto":
        return (
            torch_module.bfloat16
            if device == "cuda" and torch_module.cuda.is_bf16_supported()
            else torch_module.float32
        )
    return getattr(torch_module, name)


def encode_candidate_pair(
    tokenizer: Any,
    prompt: str,
    candidate: str,
    max_length: int,
) -> Tuple[List[int], List[bool]]:
    """Encode prompt + candidate while marking exactly the scored response."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(
        c2s_target(candidate), add_special_tokens=False
    )
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("C2S tokenizer must define eos_token_id")
    if not response_ids or response_ids[-1] != eos_id:
        response_ids.append(int(eos_id))
    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_length:
        raise ValueError(
            f"candidate sequence length {len(input_ids)} exceeds "
            f"--max-length {max_length}; refusing to truncate"
        )
    response_mask = [False] * len(prompt_ids) + [True] * len(response_ids)
    return input_ids, response_mask


def conditional_log_likelihoods(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    candidates: Sequence[str],
    batch_size: int,
    max_length: int,
    length_normalize: bool,
) -> np.ndarray:
    """Return a [num_cells, num_candidates] C2S response-score matrix."""
    import torch
    import torch.nn.functional as functional

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    pairs: List[Tuple[int, int, List[int], List[bool]]] = []
    for cell_index, prompt in enumerate(prompts):
        for candidate_index, candidate in enumerate(candidates):
            input_ids, response_mask = encode_candidate_pair(
                tokenizer=tokenizer,
                prompt=prompt,
                candidate=candidate,
                max_length=max_length,
            )
            pairs.append(
                (
                    cell_index,
                    candidate_index,
                    input_ids,
                    response_mask,
                )
            )

    scores = np.full((len(prompts), len(candidates)), -np.inf, dtype=np.float64)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        padded_length = max(len(item[2]) for item in batch)
        batch_input_ids: List[List[int]] = []
        batch_attention: List[List[int]] = []
        batch_response_mask: List[List[bool]] = []
        for _, _, input_ids, response_mask in batch:
            pad_count = padded_length - len(input_ids)
            batch_input_ids.append(input_ids + [int(pad_id)] * pad_count)
            batch_attention.append([1] * len(input_ids) + [0] * pad_count)
            batch_response_mask.append(
                response_mask + [False] * pad_count
            )
        input_tensor = torch.tensor(
            batch_input_ids, dtype=torch.long, device=device
        )
        attention_tensor = torch.tensor(
            batch_attention, dtype=torch.long, device=device
        )
        response_tensor = torch.tensor(
            batch_response_mask, dtype=torch.bool, device=device
        )

        with torch.inference_mode():
            logits = model(
                input_ids=input_tensor,
                attention_mask=attention_tensor,
                use_cache=False,
            ).logits
        shifted_logits = logits[:, :-1, :].float()
        shifted_labels = input_tensor[:, 1:]
        scored_tokens = response_tensor[:, 1:] & attention_tensor[:, 1:].bool()
        token_losses = functional.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            reduction="none",
        ).reshape(shifted_labels.shape)
        sequence_log_probs = -(token_losses * scored_tokens).sum(dim=1)
        token_counts = scored_tokens.sum(dim=1)
        if torch.any(token_counts == 0):
            raise RuntimeError("a candidate response contained no scored tokens")
        if length_normalize:
            sequence_log_probs = sequence_log_probs / token_counts

        for item, value in zip(batch, sequence_log_probs.cpu().tolist()):
            cell_index, candidate_index = item[:2]
            scores[cell_index, candidate_index] = float(value)
    return scores


def _validate_resume_prefix(
    source_rows: Sequence[Dict[str, Any]],
    completed_rows: Sequence[Dict[str, Any]],
) -> None:
    if len(completed_rows) > len(source_rows):
        raise ValueError("resume output is longer than the input")
    for index, completed in enumerate(completed_rows):
        if completed.get("id") != source_rows[index].get("id"):
            raise ValueError(
                f"resume prefix mismatch at row {index}: "
                f"{completed.get('id')!r} != {source_rows[index].get('id')!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--organism", default="Homo sapiens")
    parser.add_argument("--sum-log-probs", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device == "auto":
        device = "cpu"
    dtype = resolve_dtype(torch, args.dtype, device)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    records = read_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    results: List[Dict[str, Any]] = []
    if args.resume and args.output.is_file():
        results = read_records(args.output)
        _validate_resume_prefix(records, results)
        print(f"Resuming {args.output} after {len(results)} rows", flush=True)

    for record_index in range(len(results), len(records)):
        row = records[record_index]
        _, cells, candidates = parse_cellpuzzles_prompt(row["user_msg"])
        prompts = [
            build_c2s_prompt(
                genes=genes,
                candidates=candidates,
                organism=args.organism,
            )
            for genes in cells
        ]
        score_matrix = conditional_log_likelihoods(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            candidates=candidates,
            batch_size=args.batch_size,
            max_length=args.max_length,
            length_normalize=not args.sum_log_probs,
        )
        labels, candidate_indices = select_independent(
            score_matrix, candidates
        )
        updated = dict(row)
        updated["prediction"] = f"<answer>{' | '.join(labels)}</answer>"
        updated["c2s_protocol"] = "closed_set_cell_level_independent"
        updated["c2s_prompt_version"] = C2S_PROMPT_VERSION
        updated["c2s_candidate_indices"] = candidate_indices
        # Keep full float precision so the saved top-k ranking cannot disagree
        # with the argmax used to construct predictions on near-tied labels.
        updated["c2s_score_matrix"] = score_matrix.tolist()
        updated["c2s_length_normalized"] = not args.sum_log_probs
        results.append(updated)

        completed = record_index + 1
        if (
            completed % args.save_every == 0
            or completed == len(records)
        ):
            write_json(args.output, results)
            print(
                f"[{completed}/{len(records)}] saved {args.output}",
                flush=True,
            )

    print(
        json.dumps(
            {
                "model": args.model,
                "input": str(args.input),
                "output": str(args.output),
                "rows": len(results),
                "device": device,
                "dtype": str(dtype),
                "protocol": "closed_set_cell_level_independent",
                "length_normalized": not args.sum_log_probs,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
