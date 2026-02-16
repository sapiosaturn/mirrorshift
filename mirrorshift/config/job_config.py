"""Dataclass configuration schema for mirrorshift."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ScheduleName = Literal[
    "linear_warmup",
    "wsd_exponential",
    "wsd_linear",
    "wsd_cosine",
]
DeviceName = Literal["cpu", "cuda"]

DEFAULT_TRAIN_CONFIG = "mirrorshift/config/train_configs/small.toml"


@dataclass(frozen=True)
class Job:
    config_file: str = DEFAULT_TRAIN_CONFIG
    print_config: bool = True


@dataclass(frozen=True)
class Run:
    dataset: str = "mirrorshift/datasets/coqa_stories.txt"
    spec: str = "causal_lm"
    log_dir: str = "runs"


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50281
    num_layers: int = 2
    num_kv_heads: int = 4
    embedding_dim: int = 128
    num_heads: int = 8
    context_length: int = 64
    feedforward_dim: int = 384
    attention_type: Literal["gqa", "mla"] = "mla"
    q_lora_rank: int | None = 64
    kv_lora_rank: int | None = 64
    qk_nope_head_dim: int | None = 32
    qk_rope_head_dim: int | None = 16
    v_head_dim: int | None = 64


@dataclass(frozen=True)
class TrainingConfig:
    device: DeviceName = "cpu"
    batch_size: int = 16
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 50
    lr_schedule: ScheduleName = "wsd_exponential"
    max_steps: int = 500
    compile: bool = False
    log_every: int = 25


@dataclass(frozen=True)
class JobConfig:
    job: Job = field(default_factory=Job)
    run: Run = field(default_factory=Run)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def maybe_log(self, logger: Any) -> None:
        if self.job.print_config:
            logger.info(
                "Resolved config:\n%s",
                json.dumps(self.to_dict(), indent=2, sort_keys=True),
            )


def validate_training_config(config: TrainingConfig) -> None:
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("training.device must be 'cpu' or 'cuda'")
    if config.batch_size <= 0:
        raise ValueError("training.batch_size must be > 0")
    if config.learning_rate <= 0:
        raise ValueError("training.learning_rate must be > 0")
    if config.lr_warmup_steps < 0:
        raise ValueError("training.lr_warmup_steps must be >= 0")
    if config.max_steps <= 0:
        raise ValueError("training.max_steps must be > 0")
    if config.log_every <= 0:
        raise ValueError("training.log_every must be > 0")


def validate_model_config(config: ModelConfig) -> None:
    if config.attention_type not in {"gqa", "mla"}:
        raise ValueError("model.attention_type must be 'gqa' or 'mla'")
    if config.vocab_size <= 0:
        raise ValueError("model.vocab_size must be > 0")
    if config.num_layers <= 0:
        raise ValueError("model.num_layers must be > 0")
    if config.embedding_dim <= 0:
        raise ValueError("model.embedding_dim must be > 0")
    if config.num_heads <= 0:
        raise ValueError("model.num_heads must be > 0")
    if config.num_kv_heads <= 0:
        raise ValueError("model.num_kv_heads must be > 0")
    if config.context_length <= 0:
        raise ValueError("model.context_length must be > 0")
    if config.feedforward_dim <= 0:
        raise ValueError("model.feedforward_dim must be > 0")
    if config.embedding_dim % config.num_heads != 0:
        raise ValueError("model.embedding_dim must be divisible by num_heads")
    if config.num_heads % config.num_kv_heads != 0:
        raise ValueError("model.num_heads must be divisible by num_kv_heads")

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
                "model config for mla attention is missing: " + ", ".join(sorted(missing))
            )


def validate_run_config(config: Run) -> None:
    if not config.dataset:
        raise ValueError("run.dataset must be non-empty")
    if not config.spec:
        raise ValueError("run.spec must be non-empty")
    if not config.log_dir:
        raise ValueError("run.log_dir must be non-empty")

