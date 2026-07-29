# training/lora_trainer.py
import collections
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from training.hf_cache import (
    ensure_model_cached as _ensure_model_cached,
    resolve_cached_hf_snapshot,
    verify_hf_model_snapshot,
)
from training.hparams import (
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LORA_ALPHA_MULTIPLIER,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_MICRO_BATCH_SIZE,
    DEFAULT_WEIGHT_DECAY,
    MAX_EFFECTIVE_BATCH_SIZE,
    MAX_EPOCHS,
    MAX_LEARNING_RATE,
    MIN_EPOCHS,
    MIN_LEARNING_RATE,
    VALID_GRADIENT_ACCUMULATION_STEPS,
    VALID_LORA_DROPOUTS,
    VALID_LORA_RANKS,
    VALID_MICRO_BATCH_SIZES,
    VALID_WEIGHT_DECAYS,
)

TrainingOutput = collections.namedtuple("TrainingOutput", ["weights_ref", "gguf_path"])

SFT_LOSS_CONTRACT_VERSION = 2  # explicit assistant/completion-only labels
NON_THINKING_TEMPLATE_KWARGS = {"enable_thinking": False}


def _configured_max_seq_length() -> int:
    try:
        value = int(os.environ.get("SLM_MAX_SEQ_LENGTH", "4096"))
    except (TypeError, ValueError):
        value = 4096
    return min(max(value, 128), 32768)


def _validate_training_sequence_lengths(
    formatted_rows: list[dict],
    tokenizer,
    max_seq_length: int,
) -> None:
    """Fail before SFT rather than truncating prompt or gold completion."""
    for index, row in enumerate(formatted_rows):
        input_ids = row.get("input_ids")
        if input_ids is None:
            encoded = tokenizer(
                row["text"],
                truncation=False,
                add_special_tokens=True,
            )
            input_ids = (
                encoded.get("input_ids")
                if hasattr(encoded, "get")
                else None
            )
        if not isinstance(input_ids, (list, tuple)):
            # Lightweight test doubles may not expose concrete token IDs.
            continue
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            input_ids = input_ids[0]
        if not all(isinstance(token, int) for token in input_ids):
            continue
        token_count = len(input_ids)
        if token_count > max_seq_length:
            raise ValueError(
                f"Formatted training row {index} contains {token_count} tokens, "
                f"exceeding configured context {max_seq_length}. Refusing to "
                "silently truncate target-critical prompt or completion content."
            )


class CompletionOnlyDataCollator:
    """Right-pad tokenized rows and mask every non-assistant label.

    SFTTrainer accepts ordinary callable collators for already-tokenized datasets.
    Keeping the completion mask explicit avoids relying on model-specific chat-template
    generation-mask extensions, which are not uniformly available for Qwen text-only
    training through FastVisionModel.
    """

    def __init__(self, pad_token_id: int):
        if isinstance(pad_token_id, bool) or not isinstance(
            pad_token_id, int
        ):
            raise ValueError(
                f"pad_token_id must be an integer, got {pad_token_id!r}"
            )
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]):
        import torch

        if not features:
            raise ValueError("completion-only collator received an empty batch")
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_rows = []
        attention_rows = []
        label_rows = []
        for row_index, feature in enumerate(features):
            input_ids = list(feature["input_ids"])
            completion_mask = list(feature["completion_mask"])
            if len(input_ids) != len(completion_mask):
                raise ValueError(
                    f"row {row_index} input_ids/completion_mask length mismatch"
                )
            if not input_ids or 1 not in completion_mask:
                raise ValueError(
                    f"row {row_index} has no assistant target tokens"
                )
            if any(value not in (0, 1) for value in completion_mask):
                raise ValueError(
                    f"row {row_index} completion_mask must contain only 0/1"
                )
            padding = max_length - len(input_ids)
            input_rows.append(
                input_ids + [self.pad_token_id] * padding
            )
            attention_rows.append(
                [1] * len(input_ids) + [0] * padding
            )
            label_rows.append(
                [
                    token if is_completion else -100
                    for token, is_completion in zip(
                        input_ids,
                        completion_mask,
                    )
                ]
                + [-100] * padding
            )
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_rows,
                dtype=torch.long,
            ),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }


def _concrete_token_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        truncation=False,
        add_special_tokens=True,
    )
    input_ids = (
        encoded.get("input_ids")
        if hasattr(encoded, "get")
        else None
    )
    if (
        isinstance(input_ids, (list, tuple))
        and input_ids
        and isinstance(input_ids[0], (list, tuple))
    ):
        input_ids = input_ids[0]
    if not isinstance(input_ids, (list, tuple)) or not all(
        isinstance(token, int) and not isinstance(token, bool)
        for token in input_ids
    ):
        raise TypeError(
            "training tokenizer must return concrete integer input_ids"
        )
    return list(input_ids)


def _training_turn(
    example: dict,
    task_type: str,
    classification_labels: list[str],
) -> tuple[str, str, str]:
    """Return (user prompt, assistant target, plain-text fallback marker)."""
    if task_type == "classification":
        from eval.scorers.classification import build_classify_prompt

        return (
            build_classify_prompt(
                example.get("text", ""),
                classification_labels,
            ),
            str(example.get("label", "")),
            "Answer",
        )
    if task_type == "NER":
        return (
            "Extract named entities from the text. "
            "Reply with a JSON list of objects with \"text\" and \"type\" "
            "keys.\n\n"
            f"Text: {example['text']}",
            json.dumps(example.get("entities", [])),
            "Entities",
        )
    if task_type == "code_generation":
        from eval.scorers.generation import build_code_prompt

        return (
            build_code_prompt(example),
            build_code_training_target(example),
            "Answer",
        )
    if task_type in ("generation", "math_reasoning"):
        user_message = example.get(
            "text",
            example.get("prompt", ""),
        )
        raw_answer = example.get(
            "answer",
            example.get(
                "response",
                example.get("label", ""),
            ),
        )
        cot = example.get("cot_reasoning", "")
        assistant_message = (
            f"<reasoning>\n{cot}\n</reasoning>\n\n{raw_answer}"
            if cot
            else raw_answer
        )
        return (
            str(user_message),
            str(assistant_message),
            "Answer",
        )
    raise ValueError(
        f"completion-only SFT does not support task_type={task_type!r}"
    )


def _build_completion_only_rows(
    raw_rows: list[dict],
    tokenizer,
    task_type: str,
) -> list[dict]:
    """Render exact train/serve chat text and build assistant completion masks."""
    has_chat_template = (
        hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template is not None
    )
    if not has_chat_template and "qwen" in str(
        getattr(tokenizer, "name_or_path", "")
    ).lower():
        raise RuntimeError(
            "Qwen tokenizer has no chat template; cannot enforce non-thinking "
            "train/inference parity."
        )
    labels = (
        sorted({
            str(example.get("label", ""))
            for example in raw_rows
            if example.get("label")
        })
        if task_type == "classification"
        else []
    )
    rows = []
    for row_index, example in enumerate(raw_rows):
        user_message, assistant_message, fallback_marker = _training_turn(
            example,
            task_type,
            labels,
        )
        if not assistant_message:
            raise ValueError(
                f"training row {row_index} has an empty assistant target"
            )
        if has_chat_template:
            prompt_text = apply_non_thinking_chat_template(
                tokenizer,
                [{"role": "user", "content": user_message}],
                add_generation_prompt=True,
            )
            full_text = apply_non_thinking_chat_template(
                tokenizer,
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ],
                add_generation_prompt=False,
            )
        else:
            prompt_text = f"{user_message}\n\n{fallback_marker}: "
            full_text = prompt_text + assistant_message

        prompt_ids = _concrete_token_ids(tokenizer, prompt_text)
        full_ids = _concrete_token_ids(tokenizer, full_text)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                f"training row {row_index} prompt tokenization is not a prefix "
                "of the full chat turn; refusing an ambiguous assistant-loss "
                "boundary"
            )
        completion_length = len(full_ids) - len(prompt_ids)
        if completion_length <= 0:
            raise RuntimeError(
                f"training row {row_index} produced no assistant target tokens"
            )
        rows.append({
            "text": full_text,
            "input_ids": full_ids,
            "completion_mask": (
                [0] * len(prompt_ids)
                + [1] * completion_length
            ),
        })
    return rows


def _load_training_model(
    config: "TrainingConfig",
    max_seq_length: int,
):
    """Load a fresh base model, attach exactly one fresh LoRA adapter, and tokenizer."""
    from unsloth import FastLanguageModel

    multimodal = is_multimodal_model(config.base_model)
    if multimodal:
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=config.base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=config.lora_rank is not None,
            trust_remote_code=True,
        )
        if config.lora_rank is not None:
            model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
            )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=config.lora_rank is not None,
            trust_remote_code=True,
        )
        if config.lora_rank is not None:
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                bias="none",
            )
    return model, text_tokenizer(tokenizer)


def _completion_collator_for(tokenizer) -> CompletionOnlyDataCollator:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise RuntimeError(
            "training tokenizer must expose an integer pad_token_id or "
            "eos_token_id for completion-only collation"
        )
    return CompletionOnlyDataCollator(pad_token_id)


def build_code_training_target(example: dict) -> str:
    """Keep optional implementation reasoning executable and eval-compatible."""
    code = str(
        example.get(
            "answer",
            example.get("response", example.get("label", "")),
        )
        or ""
    )
    reasoning = str(example.get("cot_reasoning") or "").strip()
    if not reasoning:
        return code
    comments = "\n".join(
        f"# {line}" if line else "#"
        for line in reasoning.splitlines()
    )
    return f"{comments}\n{code}"


def is_multimodal_model(model_id: str) -> bool:
    """True if `model_id` is a multimodal ("Causal LM with Vision") pool entry.

    Looked up from ANDROID_POOL's `multimodal` flag so training/inference can pick the
    FastVisionModel path without threading a flag through every call site. Falls back to
    False (plain text load) if the pool can't be imported or the id isn't found.
    """
    try:
        from config.android_pool import ANDROID_POOL
        return any(m.model_id == model_id and getattr(m, "multimodal", False) for m in ANDROID_POOL)
    except Exception:
        return False


def text_tokenizer(tok):
    """Return the underlying TEXT tokenizer for a possibly-multimodal load.

    Multimodal models (e.g. Qwen3.5, Gemma-3n) load as a *processor* that wraps a
    text tokenizer plus an image/video processor. Calling `apply_chat_template` /
    tokenizing text through the processor routes the text into the vision path and
    crashes ("Incorrect image source ... Got <|im_start|>user", BUGS B123). The text
    tokenizer is exposed as processor.tokenizer — use it directly for text-only LoRA
    so the vision processor is never involved. For plain text models this is a no-op.
    """
    # Only unwrap for an actual multimodal *Processor* class (e.g. Qwen2VLProcessor,
    # Gemma3Processor). Gate on the class name so a plain text tokenizer — or a test
    # MagicMock, which auto-creates a `.tokenizer` attribute — is never unwrapped.
    if type(tok).__name__.endswith("Processor"):
        inner = getattr(tok, "tokenizer", None)
        if inner is not None and hasattr(inner, "apply_chat_template"):
            return inner
    return tok


def apply_non_thinking_chat_template(
    tokenizer,
    messages,
    *,
    add_generation_prompt: bool,
):
    """Render target-SLM chat consistently with Qwen thinking disabled."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **NON_THINKING_TEMPLATE_KWARGS,
    )


@dataclass
class TrainingConfig:
    base_model: str
    nr_epochs: int
    learning_rate: float
    batch_size: int | None = None  # legacy alias for micro_batch_size
    lora_rank: int | None = None  # None = full fine-tune
    task_type: str = "classification"
    lora_alpha: int | None = None
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    micro_batch_size: int | None = None
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    effective_batch_size: int | None = None

    def __post_init__(self):
        if self.lora_rank is not None and (
            isinstance(self.lora_rank, bool)
            or not isinstance(self.lora_rank, int)
            or self.lora_rank not in VALID_LORA_RANKS
        ):
            raise ValueError(
                f"lora_rank must be one of {VALID_LORA_RANKS}, "
                f"got {self.lora_rank}"
            )
        if self.lora_rank is None:
            if self.lora_alpha is not None:
                raise ValueError(
                    "lora_alpha must be None when lora_rank is None"
                )
        else:
            if self.lora_alpha is None:
                self.lora_alpha = (
                    self.lora_rank * DEFAULT_LORA_ALPHA_MULTIPLIER
                )
            valid_alphas = (
                self.lora_rank,
                self.lora_rank * 2,
                self.lora_rank * 4,
            )
            if (
                isinstance(self.lora_alpha, bool)
                or not isinstance(self.lora_alpha, int)
                or self.lora_alpha not in valid_alphas
            ):
                raise ValueError(
                    f"lora_alpha must be one of {valid_alphas} for "
                    f"lora_rank={self.lora_rank}, got {self.lora_alpha}"
                )

        def canonical_float(value, allowed, field_name):
            if isinstance(value, bool):
                raise ValueError(
                    f"{field_name} must be one of {allowed}, got {value!r}"
                )
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{field_name} must be one of {allowed}, got {value!r}"
                ) from exc
            match = next(
                (
                    candidate
                    for candidate in allowed
                    if math.isclose(
                        number,
                        candidate,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ),
                None,
            )
            if match is None:
                raise ValueError(
                    f"{field_name} must be one of {allowed}, got {value!r}"
                )
            return float(match)

        self.lora_dropout = canonical_float(
            self.lora_dropout,
            VALID_LORA_DROPOUTS,
            "lora_dropout",
        )
        self.weight_decay = canonical_float(
            self.weight_decay,
            VALID_WEIGHT_DECAYS,
            "weight_decay",
        )
        if isinstance(self.nr_epochs, bool) or not isinstance(
            self.nr_epochs, int
        ):
            raise ValueError(
                f"nr_epochs must be an integer in [{MIN_EPOCHS}, "
                f"{MAX_EPOCHS}], got {self.nr_epochs!r}"
            )
        if not (MIN_EPOCHS <= self.nr_epochs <= MAX_EPOCHS):
            raise ValueError(
                f"nr_epochs must be in [{MIN_EPOCHS}, {MAX_EPOCHS}], "
                f"got {self.nr_epochs}"
            )
        try:
            self.learning_rate = float(self.learning_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "learning_rate must be numeric in "
                f"[{MIN_LEARNING_RATE}, {MAX_LEARNING_RATE}]"
            ) from exc
        if not (
            math.isfinite(self.learning_rate)
            and MIN_LEARNING_RATE
            <= self.learning_rate
            <= MAX_LEARNING_RATE
        ):
            raise ValueError(
                f"learning_rate must be in [{MIN_LEARNING_RATE}, "
                f"{MAX_LEARNING_RATE}], got {self.learning_rate}"
            )

        if (
            self.batch_size is not None
            and self.micro_batch_size is not None
            and self.batch_size != self.micro_batch_size
        ):
            raise ValueError(
                "legacy batch_size and micro_batch_size must agree when both "
                f"are provided, got {self.batch_size} and "
                f"{self.micro_batch_size}"
            )
        micro_batch = (
            self.micro_batch_size
            if self.micro_batch_size is not None
            else self.batch_size
        )
        if micro_batch is None:
            micro_batch = DEFAULT_MICRO_BATCH_SIZE
        if (
            isinstance(micro_batch, bool)
            or not isinstance(micro_batch, int)
            or micro_batch not in VALID_MICRO_BATCH_SIZES
        ):
            raise ValueError(
                f"micro_batch_size must be one of "
                f"{VALID_MICRO_BATCH_SIZES}, got {micro_batch}"
            )
        if (
            isinstance(self.gradient_accumulation_steps, bool)
            or not isinstance(self.gradient_accumulation_steps, int)
            or self.gradient_accumulation_steps
            not in VALID_GRADIENT_ACCUMULATION_STEPS
        ):
            raise ValueError(
                "gradient_accumulation_steps must be one of "
                f"{VALID_GRADIENT_ACCUMULATION_STEPS}, got "
                f"{self.gradient_accumulation_steps}"
            )
        self.micro_batch_size = int(micro_batch)
        self.batch_size = self.micro_batch_size
        self.gradient_accumulation_steps = int(
            self.gradient_accumulation_steps
        )
        derived_effective_batch_size = (
            self.micro_batch_size
            * self.gradient_accumulation_steps
        )
        if derived_effective_batch_size > MAX_EFFECTIVE_BATCH_SIZE:
            raise ValueError(
                f"derived effective_batch_size={derived_effective_batch_size} "
                f"exceeds {MAX_EFFECTIVE_BATCH_SIZE}"
            )
        if self.effective_batch_size is not None:
            if (
                isinstance(self.effective_batch_size, bool)
                or not isinstance(self.effective_batch_size, int)
                or self.effective_batch_size
                != derived_effective_batch_size
            ):
                raise ValueError(
                    f"effective_batch_size={self.effective_batch_size!r} "
                    f"must equal the derived value "
                    f"{self.micro_batch_size}*"
                    f"{self.gradient_accumulation_steps}="
                    f"{derived_effective_batch_size}"
                )
        self.effective_batch_size = derived_effective_batch_size

def _run_unsloth_training(
    dataset_path: str,
    config: TrainingConfig,
    output_dir: str,
    task_type: str = "classification",
) -> str:
    """
    Run LoRA fine-tuning via Unsloth.
    Returns the path to the saved checkpoint directory.
    task_type controls the prompt format:
      - "classification": label classification prompt
      - "NER": entity extraction prompt
      - "generation": text-to-answer prompt
    """
    import torch
    from agent.logging_setup import quiet_ml_logging
    quiet_ml_logging()

    max_seq_length = _configured_max_seq_length()
    _ensure_model_cached(config.base_model)

    model, tokenizer = _load_training_model(config, max_seq_length)
    # Unsloth must patch TRL before SFTTrainer is imported.
    from trl import SFTTrainer

    # Load dataset
    with open(dataset_path, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    # Format training examples using the model's chat template when available.
    # This achieves train/serve parity (paper Appendix D.1) and structurally
    # separates the prompt from the completion so loss masking works correctly.
    has_chat_template = (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    )
    if not has_chat_template and "qwen" in config.base_model.lower():
        raise RuntimeError(
            f"Qwen tokenizer for {config.base_model!r} has no chat template; "
            "cannot enforce non-thinking train/inference parity."
        )

    _formatted = _build_completion_only_rows(
        raw,
        tokenizer,
        task_type,
    )
    _validate_training_sequence_lengths(
        _formatted,
        tokenizer,
        max_seq_length,
    )

    # Early stopping / best-checkpoint (B161 Addition 2): carve a small VALIDATION split from
    # the TRAINING data (never the held-out eval set), evaluate periodically, and keep the
    # best-val checkpoint instead of the (often overfit) final-epoch one. LoRA memorizes tiny
    # datasets fast, so this reduces overfit + score oscillation. Guarded: mid-training saving
    # can hit an Unsloth/SFTConfig pickle issue (B109-adjacent), so the whole early-stopping
    # path falls back to the plain final-checkpoint path on ANY failure (see below).
    import os as _os
    _early_stop = _os.environ.get("SLM_EARLY_STOPPING", "1") != "0"
    _val_frac = float(_os.environ.get("SLM_VAL_FRACTION", "0.12"))
    _min_for_val = int(_os.environ.get("SLM_MIN_FOR_VAL", "60"))
    _use_val = _early_stop and len(_formatted) >= _min_for_val
    _val_rows = []
    if _use_val:
        import random as _rnd
        _rng = _rnd.Random(1234)
        _idx = list(range(len(_formatted)))
        _rng.shuffle(_idx)
        _n_val = max(8, int(len(_formatted) * _val_frac))
        _val_ids = set(_idx[:_n_val])
        _train_rows = [_formatted[i] for i in _idx[_n_val:]]
        _val_rows = [_formatted[i] for i in _idx[:_n_val]]
        _formatted = _train_rows

    from datasets import Dataset as _DS
    dataset = _DS.from_list(_formatted)
    eval_dataset = _DS.from_list(_val_rows) if _val_rows else None
    data_collator = _completion_collator_for(tokenizer)

    from trl import SFTConfig
    _bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # trl >= ~0.20 moved the SFT-specific knobs into SFTConfig: dataset_text_field and
    # max_length live there (max_seq_length was renamed to max_length), and the tokenizer
    # is passed as processing_class. The previous code passed max_seq_length/tokenizer as
    # bare SFTTrainer kwargs, which the current trl silently ignores — so NO truncation was
    # applied. Long (math CoT / NER abstract) sequences then reached Unsloth's fused
    # cross-entropy with mismatched logits vs label lengths and crashed (see BUGS B109).
    # Build an SFTConfig with the explicit shared 4096-token max_length. Preflight above
    # rejects oversized rows instead of silently truncating prompt or target content.
    # Epochs act as a CEILING when early stopping is on (train longer, stop at best val).
    # The searched epoch count is the actual ceiling. Raising every candidate to
    # the same environment default would make epochs 1..7 false identities.
    _epoch_ceiling = config.nr_epochs

    def _build_sft_config(with_eval: bool):
        common = dict(
            output_dir=output_dir,
            num_train_epochs=(_epoch_ceiling if with_eval else config.nr_epochs),
            per_device_train_batch_size=config.micro_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            # Always wire the selected optimizer regularization, including the
            # no-validation fallback path.
            weight_decay=config.weight_decay,
            fp16=not _bf16 and torch.cuda.is_available(),
            bf16=_bf16,
            logging_steps=10,
            disable_tqdm=True,
            report_to="none",
            dataset_text_field="text",
            max_length=max_seq_length,
            # The dataset is pre-tokenized with an explicit completion_mask and
            # a custom collator that writes -100 over every prompt token.
            completion_only_loss=True,
        )
        if with_eval:
            # Evaluate + checkpoint periodically; keep the best-val model. weight_decay +
            # dropout guard against overfit on the small set (per the epochs research).
            common.update(
                eval_strategy="steps",
                eval_steps=int(_os.environ.get("SLM_EVAL_STEPS", "20")),
                save_strategy="steps",
                save_steps=int(_os.environ.get("SLM_EVAL_STEPS", "20")),
                save_total_limit=1,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            )
        else:
            # "no" avoids mid-training checkpointing (Unsloth/SFTConfig pickle issue, B109).
            common.update(save_strategy="no")
        return SFTConfig(**common)

    def _make_trainer(
        cfg,
        with_eval: bool,
        trainer_model,
        train_dataset,
        validation_dataset,
        processing_tokenizer,
        completion_collator,
    ):
        cbs = []
        if with_eval:
            from transformers import EarlyStoppingCallback
            cbs = [EarlyStoppingCallback(
                early_stopping_patience=int(_os.environ.get("SLM_EARLY_STOP_PATIENCE", "3")))]
        kw = dict(
            model=trainer_model,
            train_dataset=train_dataset,
            args=cfg,
            data_collator=completion_collator,
        )
        if with_eval and validation_dataset is not None:
            kw["eval_dataset"] = validation_dataset
        if cbs:
            kw["callbacks"] = cbs
        try:
            return SFTTrainer(
                processing_class=processing_tokenizer,
                **kw,
            )
        except TypeError:
            return SFTTrainer(
                tokenizer=processing_tokenizer,
                dataset_text_field="text",
                **kw,
            )

    if _use_val:
        print(f"[train] early stopping ON: val={len(_val_rows)} examples, epoch ceiling "
              f"{_epoch_ceiling}, eval/save every {_os.environ.get('SLM_EVAL_STEPS','20')} steps, "
              f"patience {_os.environ.get('SLM_EARLY_STOP_PATIENCE','3')} → keep best-val checkpoint; "
              f"micro_batch={config.micro_batch_size} grad_accum={config.gradient_accumulation_steps} "
              f"effective_batch={config.effective_batch_size} weight_decay={config.weight_decay}")
    trainer = None
    early_stop_failure = None
    try:
        trainer = _make_trainer(
            _build_sft_config(_use_val),
            _use_val,
            model,
            dataset,
            eval_dataset,
            tokenizer,
            data_collator,
        )
        trainer.train()
    except Exception as exc:  # noqa: BLE001
        if not _use_val:
            raise
        # Capture immutable diagnostics only. Cleanup and reload MUST happen after
        # this except suite exits: while `exc` is bound, its traceback can retain
        # trainer frames and CUDA tensors.
        early_stop_failure = (
            f"{type(exc).__name__}: {str(exc)[:120]}"
        )

    if early_stop_failure is not None:
        # A failed trainer may already have stepped the optimizer. Reusing either
        # that model or the reduced train split would realize a different H than
        # the one recorded by the DAG. We are now outside the exception scope, so
        # the exception/traceback has been cleared before releasing the failed
        # object graph, collecting cycles, emptying CUDA, and reloading.
        print(
            f"[train] ⚠ early-stopping path failed ({early_stop_failure}); "
            "reloading fresh base+LoRA and retrying all original rows "
            "(no val/early-stop)"
        )
        failed_stack = [trainer, model, tokenizer]
        trainer = None
        model = None
        tokenizer = None
        dataset = None
        eval_dataset = None
        data_collator = None
        # Clearing the shared container also removes the caller-frame references
        # before gc.collect() runs.
        failed_stack.clear()
        del failed_stack
        import gc as _gc

        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _purge_trainer_checkpoints(output_dir)

        model, tokenizer = _load_training_model(
            config,
            max_seq_length,
        )
        if (
            not getattr(tokenizer, "chat_template", None)
            and "qwen" in config.base_model.lower()
        ):
            raise RuntimeError(
                f"Qwen tokenizer for {config.base_model!r} has no chat "
                "template after fresh fallback reload"
            )
        fallback_rows = _build_completion_only_rows(
            raw,
            tokenizer,
            task_type,
        )
        _validate_training_sequence_lengths(
            fallback_rows,
            tokenizer,
            max_seq_length,
        )
        dataset = _DS.from_list(fallback_rows)
        data_collator = _completion_collator_for(tokenizer)
        _use_val = False
        trainer = _make_trainer(
            _build_sft_config(False),
            False,
            model,
            dataset,
            None,
            tokenizer,
            data_collator,
        )
        trainer.train()

    checkpoint_path = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)

    # The HF Trainer's `checkpoint-*` dirs are mid-training resume state (optimizer.pt +
    # a duplicate adapter + a duplicate tokenizer.json, ~1.2 GB/iteration). They are dead
    # now that final_checkpoint exists: load_best_model_at_end has already pulled the
    # best-val weights into the model we just saved. Purging only AFTER a successful save
    # keeps resume state on disk if training crashed part-way.
    _purge_trainer_checkpoints(output_dir)

    # Free the training model's VRAM before the caller loads the checkpoint for eval.
    # Without this, the just-trained model stays resident and, combined with the
    # inference load, pushes large (4B) models into CPU/meta offload → Unsloth
    # fast-generate crashes with "Invalid target device: None" (B142).
    try:
        import gc
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return checkpoint_path

def _purge_trainer_checkpoints(output_dir: str) -> None:
    """Remove the HF Trainer's `checkpoint-*` resume state under `output_dir`.

    Each holds `optimizer.pt`, a duplicate adapter and a duplicate 20 MB
    `tokenizer.json` — roughly 1.2 GB per iteration that exists solely to resume an
    interrupted `trainer.train()`. `final_checkpoint/` is never a match (it does not
    carry the `checkpoint-` prefix) and neither are plain files.
    """
    import shutil as _shutil
    from pathlib import Path as _Path

    for checkpoint_dir in _Path(output_dir).glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            _shutil.rmtree(checkpoint_dir, ignore_errors=True)


def merge_for_quantization(checkpoint_path: str, output_dir: str) -> str:
    """
    Merge LoRA adapters into the base model weights and save as a full HF checkpoint.
    Required before GGUF quantization — GGUF cannot be produced from adapter-only checkpoints.
    Returns path to the merged HF checkpoint directory.
    """
    from unsloth import FastLanguageModel, FastVisionModel
    from agent.logging_setup import quiet_ml_logging
    quiet_ml_logging()
    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.isfile(adapter_config):
        try:
            with open(adapter_config, encoding="utf-8") as handle:
                adapter_data = json.load(handle)
                base_identity = adapter_data.get(
                    "base_model_name_or_path"
                )
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(
                f"Cannot read adapter base model from {adapter_config}: {exc}"
            ) from exc
        if not isinstance(base_identity, str) or not base_identity.strip():
            raise ValueError(
                f"Adapter config {adapter_config} has no valid "
                "base_model_name_or_path"
            )

        local_base = resolve_cached_hf_snapshot(base_identity)
        loader = (
            FastVisionModel
            if is_multimodal_model(base_identity)
            else FastLanguageModel
        )
        with tempfile.TemporaryDirectory(
            prefix=".local-merge-adapter-",
            dir=output_dir,
        ) as staged_adapter:
            staged_path = Path(staged_adapter)
            for source in Path(checkpoint_path).iterdir():
                if source.name == "adapter_config.json":
                    continue
                os.symlink(
                    source.resolve(),
                    staged_path / source.name,
                    target_is_directory=source.is_dir(),
                )
            adapter_data["base_model_name_or_path"] = local_base
            (staged_path / "adapter_config.json").write_text(
                json.dumps(adapter_data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            model, tokenizer = loader.from_pretrained(
                model_name=staged_adapter,
                max_seq_length=_configured_max_seq_length(),
                load_in_4bit=False,
                local_files_only=True,
                trust_remote_code=True,
            )
            # Unsloth derives the merge shard source from this field. Keep it
            # on the verified snapshot rather than the adapter's Hub identity.
            model.config._name_or_path = local_base
            model.save_pretrained_merged(
                merged_dir,
                tokenizer,
                save_method="merged_16bit",
            )
    else:
        loader = (
            FastVisionModel
            if is_multimodal_model(checkpoint_path)
            else FastLanguageModel
        )
        model, tokenizer = loader.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=_configured_max_seq_length(),
            load_in_4bit=False,
            trust_remote_code=True,
        )
        model.save_pretrained_merged(
            merged_dir,
            tokenizer,
            save_method="merged_16bit",
        )
    verify_hf_model_snapshot(merged_dir)
    return merged_dir


def run_lora_training(
    dataset_path: str,
    config: TrainingConfig,
    output_dir: str = "artifacts",
    task_type: str | None = None,
) -> TrainingOutput:
    """
    Train model with LoRA (or full fine-tune if lora_rank is None).
    Returns TrainingOutput(weights_ref, gguf_path=None).
    weights_ref is a path string usable by infer() and infer_batch().
    gguf_path is always None here — set by the quantize step in evaluate_node.
    Always trains from the base model, never from a prior checkpoint.
    task_type defaults to config.task_type if not provided.
    """
    os.makedirs(output_dir, exist_ok=True)
    effective_task_type = task_type if task_type is not None else config.task_type
    checkpoint_path = _run_unsloth_training(dataset_path, config, output_dir, task_type=effective_task_type)
    return TrainingOutput(weights_ref=checkpoint_path, gguf_path=None)
