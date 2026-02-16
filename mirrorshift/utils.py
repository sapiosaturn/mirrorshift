"""Configuration and scheduling utilities."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

ScheduleName = Literal[
    "linear_warmup",
    "wsd_exponential",
    "wsd_linear",
    "wsd_cosine",
]
DeviceName = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    num_layers: int
    num_kv_heads: int
    embedding_dim: int
    num_heads: int
    context_length: int
    feedforward_dim: int
    attention_type: Literal["gqa", "mla"]
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    device: DeviceName
    batch_size: int
    learning_rate: float
    lr_warmup_steps: int
    lr_schedule: ScheduleName
    max_steps: int
    compile: bool
    log_every: int


def config_to_dict(config: ModelConfig | TrainingConfig) -> dict[str, object]:
    return asdict(config)


def read_training_config(config_path: str | Path) -> TrainingConfig:
    with Path(config_path).open("r") as file:
        config_data = json.load(file)
    required_keys = {
        "device",
        "batch_size",
        "learning_rate",
        "lr_warmup_steps",
        "lr_schedule",
        "max_steps",
        "compile",
        "log_every",
    }
    missing_keys = required_keys - config_data.keys()
    if missing_keys:
        raise ValueError(
            f"Missing configuration options in {config_path}: {', '.join(sorted(missing_keys))}"
        )

    config = TrainingConfig(
        device=config_data["device"],
        batch_size=config_data["batch_size"],
        learning_rate=config_data["learning_rate"],
        lr_warmup_steps=config_data["lr_warmup_steps"],
        lr_schedule=config_data["lr_schedule"],
        max_steps=config_data["max_steps"],
        compile=config_data["compile"],
        log_every=config_data["log_every"],
    )
    validate_training_config(config)
    return config


def read_model_config(config_path: str | Path) -> ModelConfig:
    with Path(config_path).open("r") as file:
        config_data = json.load(file)
    required_keys = {
        "vocab_size",
        "num_layers",
        "num_kv_heads",
        "embedding_dim",
        "num_heads",
        "context_length",
        "feedforward_dim",
        "attention_type",
    }
    missing_keys = required_keys - config_data.keys()
    if missing_keys:
        raise ValueError(
            f"Missing configuration options in {config_path}: {', '.join(sorted(missing_keys))}"
        )
    config = ModelConfig(
        vocab_size=config_data["vocab_size"],
        num_layers=config_data["num_layers"],
        num_kv_heads=config_data["num_kv_heads"],
        embedding_dim=config_data["embedding_dim"],
        num_heads=config_data["num_heads"],
        context_length=config_data["context_length"],
        feedforward_dim=config_data["feedforward_dim"],
        attention_type=config_data["attention_type"],
        q_lora_rank=config_data.get("q_lora_rank"),
        kv_lora_rank=config_data.get("kv_lora_rank"),
        qk_nope_head_dim=config_data.get("qk_nope_head_dim"),
        qk_rope_head_dim=config_data.get("qk_rope_head_dim"),
        v_head_dim=config_data.get("v_head_dim"),
    )
    validate_model_config(config)
    return config


def validate_training_config(config: TrainingConfig) -> None:
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("training_config.device must be 'cpu' or 'cuda'")
    if config.batch_size <= 0:
        raise ValueError("training_config.batch_size must be > 0")
    if config.learning_rate <= 0:
        raise ValueError("training_config.learning_rate must be > 0")
    if config.lr_warmup_steps < 0:
        raise ValueError("training_config.lr_warmup_steps must be >= 0")
    if config.max_steps <= 0:
        raise ValueError("training_config.max_steps must be > 0")
    if config.log_every <= 0:
        raise ValueError("training_config.log_every must be > 0")


def validate_model_config(config: ModelConfig) -> None:
    if config.attention_type not in {"gqa", "mla"}:
        raise ValueError("model_config.attention_type must be 'gqa' or 'mla'")
    if config.vocab_size <= 0:
        raise ValueError("model_config.vocab_size must be > 0")
    if config.num_layers <= 0:
        raise ValueError("model_config.num_layers must be > 0")
    if config.embedding_dim <= 0:
        raise ValueError("model_config.embedding_dim must be > 0")
    if config.num_heads <= 0:
        raise ValueError("model_config.num_heads must be > 0")
    if config.num_kv_heads <= 0:
        raise ValueError("model_config.num_kv_heads must be > 0")
    if config.context_length <= 0:
        raise ValueError("model_config.context_length must be > 0")
    if config.feedforward_dim <= 0:
        raise ValueError("model_config.feedforward_dim must be > 0")
    if config.embedding_dim % config.num_heads != 0:
        raise ValueError("model_config.embedding_dim must be divisible by num_heads")
    if config.num_heads % config.num_kv_heads != 0:
        raise ValueError("model_config.num_heads must be divisible by num_kv_heads")

    if config.attention_type == "mla":
        mla_fields = {
            "q_lora_rank": config.q_lora_rank,
            "kv_lora_rank": config.kv_lora_rank,
            "qk_nope_head_dim": config.qk_nope_head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "v_head_dim": config.v_head_dim,
        }
        missing = [name for name, value in mla_fields.items() if value is None]
        if missing:
            raise ValueError(
                "model_config for mla attention is missing: "
                + ", ".join(sorted(missing))
            )


def get_lr_schedule(
    schedule: ScheduleName,
    max_lr: float,
    warmup_steps: int,
    total_steps: int,
    decay_start_factor: float = 0.8,
    start_factor: float = 0.5,
) -> Callable[[int], float]:
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")
    if max_lr <= 0:
        raise ValueError("max_lr must be > 0")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    decay_start_step = int(total_steps * decay_start_factor)
    if schedule == "linear_warmup":

        def lr_schedule(step: int) -> float:
            if step < warmup_steps and warmup_steps > 0:
                return max_lr * (
                    start_factor + (1 - start_factor) * (step / warmup_steps)
                )
            return max_lr

    elif schedule == "wsd_exponential":

        def lr_schedule(step: int) -> float:
            if step < warmup_steps and warmup_steps > 0:
                return max_lr * (
                    start_factor + (1 - start_factor) * (step / warmup_steps)
                )
            if step < decay_start_step:
                return max_lr
            progress = (step - decay_start_step) / max(1, total_steps - decay_start_step)
            decay = math.exp(-5 * progress)
            return max_lr * decay

    elif schedule == "wsd_linear":

        def lr_schedule(step: int) -> float:
            if step < warmup_steps and warmup_steps > 0:
                return max_lr * (
                    start_factor + (1 - start_factor) * (step / warmup_steps)
                )
            if step < decay_start_step:
                return max_lr
            progress = (step - decay_start_step) / max(1, total_steps - decay_start_step)
            return max_lr * (1 - progress)

    elif schedule == "wsd_cosine":

        def lr_schedule(step: int) -> float:
            if step < warmup_steps and warmup_steps > 0:
                return max_lr * (
                    start_factor + (1 - start_factor) * (step / warmup_steps)
                )
            if step < decay_start_step:
                return max_lr
            progress = (step - decay_start_step) / max(1, total_steps - decay_start_step)
            return max_lr * (1 + math.cos(math.pi * progress)) / 2

    else:
        raise ValueError(f"Unknown schedule name: {schedule}")
    return lr_schedule
