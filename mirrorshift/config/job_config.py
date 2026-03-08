"""Dataclass configuration schema for mirrorshift."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
    dataset: str = "mirrorshift/datasets/example_train.parquet"
    spec: str = "causal_lm"
    log_dir: str = "runs"
    id: str | None = None
    wandb_project: str = "mirrorshift"
    wandb_entity: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    manifest_file: str = "manifest.json"
    config_snapshot_file: str = "config.json"


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
class DataConfig:
    text_column: str = "text"
    tokenizer_name: str = "p50k_base"
    max_tokens_per_shard: int = 200_000
    plan_stride: int | None = None
    shuffle: bool = True
    shuffle_seed: int = 0
    max_documents: int | None = None


@dataclass(frozen=True)
class CheckpointConfig:
    enable: bool = False
    folder: str = "checkpoints"
    interval: int = 100
    keep_latest_k: int = 0
    load_step: int | None = None


@dataclass(frozen=True)
class JobConfig:
    job: Job = field(default_factory=Job)
    run: Run = field(default_factory=Run)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

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
    if Path(config.dataset).suffix.lower() != ".parquet":
        raise ValueError("run.dataset must point to a parquet file")
    if not config.spec:
        raise ValueError("run.spec must be non-empty")
    if not config.log_dir:
        raise ValueError("run.log_dir must be non-empty")
    if config.id is not None and not config.id:
        raise ValueError("run.id must be non-empty when provided")
    if not config.wandb_project:
        raise ValueError("run.wandb_project must be non-empty")
    if config.wandb_entity is not None and not config.wandb_entity:
        raise ValueError("run.wandb_entity must be non-empty when provided")
    if not config.manifest_file:
        raise ValueError("run.manifest_file must be non-empty")
    if not config.config_snapshot_file:
        raise ValueError("run.config_snapshot_file must be non-empty")


def validate_checkpoint_config(config: CheckpointConfig) -> None:
    if not config.folder:
        raise ValueError("checkpoint.folder must be non-empty")
    if config.interval <= 0:
        raise ValueError("checkpoint.interval must be > 0")
    if config.keep_latest_k < 0:
        raise ValueError("checkpoint.keep_latest_k must be >= 0")
    if config.load_step is not None and config.load_step < -1:
        raise ValueError("checkpoint.load_step must be -1, >= 0, or omitted")
    if config.load_step is not None and not config.enable:
        raise ValueError("checkpoint.enable must be true when checkpoint.load_step is set")


def validate_data_config(config: DataConfig) -> None:
    if not config.text_column:
        raise ValueError("data.text_column must be non-empty")
    if not config.tokenizer_name:
        raise ValueError("data.tokenizer_name must be non-empty")
    if config.max_tokens_per_shard <= 0:
        raise ValueError("data.max_tokens_per_shard must be > 0")
    if config.plan_stride is not None and config.plan_stride <= 0:
        raise ValueError("data.plan_stride must be > 0 when provided")
    if config.max_documents is not None and config.max_documents <= 0:
        raise ValueError("data.max_documents must be > 0 when provided")
