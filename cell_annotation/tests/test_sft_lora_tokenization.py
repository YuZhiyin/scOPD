import json
from pathlib import Path

from cell_annotation.sft_lora import CellSFTDataset, CompletionCollator


class EncodingLike:
    """Small stand-in for tokenizers.Encoding."""

    def __init__(self, ids):
        self.ids = ids


class BatchEncodingLike(dict):
    """Reproduce Transformers 5.x chat-template return behavior."""

    def __init__(self, ids):
        super().__init__(input_ids=ids, attention_mask=[1] * len(ids))
        self.encodings = [EncodingLike(ids)]

    def __getitem__(self, item):
        if isinstance(item, str):
            return super().__getitem__(item)
        return self.encodings[item]


class QwenPrefixMismatchTokenizer:
    """Minimal tokenizer reproducing a chat-template prefix mismatch."""

    eos_token_id = 99

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize
        assert enable_thinking is False
        if add_generation_prompt:
            # Includes the non-thinking assistant prefill.
            return [1, 2, 3, 4]
        # A complete-message rendering that is a prefix of the generation
        # prompt made the previous longest-common-prefix extraction empty.
        return [1, 2]

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        assert text.startswith("<reasoning>")
        assert text.endswith("</answer>")
        return [10, 11, 12]


class QwenBatchEncodingTokenizer(QwenPrefixMismatchTokenizer):
    """Return BatchEncoding-like objects as Transformers 5.6 does."""

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize
        assert enable_thinking is False
        return BatchEncodingLike([1, 2, 3, 4] if add_generation_prompt else [1, 2])

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return EncodingLike([10, 11, 12])


class LlamaExplicitReasoningTokenizer:
    """A standard chat template that intentionally has no thinking argument."""

    eos_token_id = 128009

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        assert add_generation_prompt
        return [128000, 128006, 9125, 128007]

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        assert text.startswith("<reasoning>")
        assert text.endswith("</answer>")
        return [21, 22, 23]


def test_nonthinking_reasoning_appends_target_to_generation_prompt(tmp_path: Path):
    data_path = tmp_path / "train.jsonl"
    row = {
        "system_msg": "Annotate each cell.",
        "user_msg": "Cell 1: NKG7, GNLY",
        "assistant_msg": (
            "<reasoning>NKG7 and GNLY identify an NK cell.</reasoning>\n"
            "<answer>natural killer cell</answer>"
        ),
    }
    data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset = CellSFTDataset(
        data_path,
        QwenPrefixMismatchTokenizer(),
        max_length=32,
        target_mode="nonthinking_reasoning",
    )
    encoded, metadata = dataset._encode(0)

    assert encoded["input_ids"] == [1, 2, 3, 4, 10, 11, 12, 99]
    assert encoded["labels"] == [-100, -100, -100, -100, 10, 11, 12, 99]
    assert metadata["raw_prompt_length"] == 4
    assert metadata["raw_target_length"] == 4
    assert metadata["target_truncated"] == 0


def test_transformers5_batch_encoding_is_flattened_before_collation(tmp_path: Path):
    data_path = tmp_path / "train.jsonl"
    row = {
        "system_msg": "Annotate each cell.",
        "user_msg": "Cell 1: NKG7, GNLY",
        "assistant_msg": (
            "<reasoning>NKG7 and GNLY identify an NK cell.</reasoning>\n"
            "<answer>natural killer cell</answer>"
        ),
    }
    data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset = CellSFTDataset(
        data_path,
        QwenBatchEncodingTokenizer(),
        max_length=32,
        target_mode="nonthinking_reasoning",
    )
    encoded, metadata = dataset._encode(0)

    assert encoded["input_ids"] == [1, 2, 3, 4, 10, 11, 12, 99]
    assert metadata["raw_prompt_length"] == 4
    assert metadata["prompt_truncated"] == 0
    assert all(isinstance(token_id, int) for token_id in encoded["input_ids"])

    batch = CompletionCollator(pad_token_id=0)([encoded, encoded])
    assert tuple(batch["input_ids"].shape) == (2, 8)
    assert batch["input_ids"][0].tolist() == encoded["input_ids"]


def test_explicit_reasoning_does_not_pass_enable_thinking(tmp_path: Path):
    data_path = tmp_path / "train.jsonl"
    row = {
        "system_msg": "Annotate each cell with explicit reasoning.",
        "user_msg": "Cell 1: NKG7, GNLY",
        "assistant_msg": (
            "<reasoning>NKG7 and GNLY identify an NK cell.</reasoning>\n"
            "<answer>natural killer cell</answer>"
        ),
    }
    data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset = CellSFTDataset(
        data_path,
        LlamaExplicitReasoningTokenizer(),
        max_length=32,
        target_mode="explicit_reasoning",
    )
    encoded, metadata = dataset._encode(0)

    assert encoded["input_ids"] == [128000, 128006, 9125, 128007, 21, 22, 23, 128009]
    assert encoded["labels"] == [-100, -100, -100, -100, 21, 22, 23, 128009]
    assert metadata["target_truncated"] == 0
