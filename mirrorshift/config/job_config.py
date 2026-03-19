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
DTypeName = Literal["float32", "bfloat16", "float16"]
ReshardPolicyName = Literal["default", "always", "never"]
ActivationCheckpointMode = Literal["none", "full", "selective"]

DEFAULT_TRAIN_CONFIG = "mirrorshift/config/train_configs/small.toml"


@dataclass(frozen=True)
class Job:
    config_file: str = DEFAULT_TRAIN_CONFIG


@dataclass(frozen=True)
class Run:
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
    max_grad_norm: float | None = None
    lr_warmup_steps: int = 50
    lr_schedule: ScheduleName = "wsd_exponential"
    max_steps: int = 500
    log_every: int = 25


@dataclass(frozen=True)
class DebugConfig:
    seed: int | None = None
    deterministic: bool = False


@dataclass(frozen=True)
class ParallelismConfig:
    dp_replicate: int = 1
    dp_shard: int = 1
    mixed_precision_param: DTypeName = "float32"
    mixed_precision_reduce: DTypeName = "float32"
    reshard_after_forward: ReshardPolicyName = "default"
    bucket_cap_mb: int = 100


@dataclass(frozen=True)
class ActivationCheckpointConfig:
    mode: ActivationCheckpointMode = "none"
    selective_ac_option: str = "2"
    preserve_rng_state: bool = True
    determinism_check: str = "default"
    debug: bool = False
    early_stop: bool = False


@dataclass(frozen=True)
class CompileConfig:
    enable: bool = False
    backend: str = "inductor"
    fullgraph: bool = True


@dataclass(frozen=True)
class DataConfig:
    snapshot_path: str = "mirrorshift/datasets/example_train_snapshot"
    plan_path: str = "mirrorshift/datasets/example_train_plan_ctx64"


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
    debug: DebugConfig = field(default_factory=DebugConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    activation_checkpoint: ActivationCheckpointConfig = field(
        default_factory=ActivationCheckpointConfig
    )
    compile: CompileConfig = field(default_factory=CompileConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def maybe_log(self, logger: Any) -> None:
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
    if config.max_grad_norm is not None and config.max_grad_norm <= 0:
        raise ValueError("training.max_grad_norm must be > 0 when provided")
    if config.lr_warmup_steps < 0:
        raise ValueError("training.lr_warmup_steps must be >= 0")
    if config.max_steps <= 0:
        raise ValueError("training.max_steps must be > 0")
    if config.log_every <= 0:
        raise ValueError("training.log_every must be > 0")


def validate_debug_config(config: DebugConfig) -> None:
    if config.seed is not None and config.seed < 0:
        raise ValueError("debug.seed must be >= 0 when provided")


def validate_parallelism_config(config: ParallelismConfig) -> None:
    if config.dp_replicate <= 0:
        raise ValueError("parallelism.dp_replicate must be > 0")
    if config.dp_shard <= 0:
        raise ValueError("parallelism.dp_shard must be > 0")
    if config.bucket_cap_mb <= 0:
        raise ValueError("parallelism.bucket_cap_mb must be > 0")
    if config.mixed_precision_param not in {"float32", "bfloat16", "float16"}:
        raise ValueError("parallelism.mixed_precision_param has unsupported dtype")
    if config.mixed_precision_reduce not in {"float32", "bfloat16", "float16"}:
        raise ValueError("parallelism.mixed_precision_reduce has unsupported dtype")
    if config.reshard_after_forward not in {"default", "always", "never"}:
        raise ValueError(
            "parallelism.reshard_after_forward must be 'default', 'always', or 'never'"
        )


def validate_activation_checkpoint_config(config: ActivationCheckpointConfig) -> None:
    if config.mode not in {"none", "full", "selective"}:
        raise ValueError("activation_checkpoint.mode must be 'none', 'full', or 'selective'")
    if config.mode == "selective":
        if config.selective_ac_option != "op" and not config.selective_ac_option.isdigit():
            raise ValueError(
                "activation_checkpoint.selective_ac_option must be 'op' or a positive integer"
            )
        if config.selective_ac_option.isdigit() and int(config.selective_ac_option) <= 0:
            raise ValueError("activation_checkpoint.selective_ac_option must be > 0")


def validate_compile_config(config: CompileConfig) -> None:
    if not config.backend:
        raise ValueError("compile.backend must be non-empty")


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
    if not config.snapshot_path:
        raise ValueError("data.snapshot_path must be non-empty")
    if not config.plan_path:
        raise ValueError("data.plan_path must be non-empty")
