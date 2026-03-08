from pathlib import Path

import torch
import torch.nn.functional as F

from mirrordata import (
    CausalLMSequenceDataset,
    DeterministicBatchLoader,
    ParquetSnapshotConfig,
    SequencePlanSpec,
    build_sequence_plan,
    build_snapshot_from_parquet,
)
from mirrorshift.modeling.causal_transformers import CausalTransformer
from mirrorshift.config import JobConfig, ModelConfig

from .spec import TrainDataBundle, TrainSpec


def build_default_model(model_config: ModelConfig) -> torch.nn.Module:
    return CausalTransformer(model_config=model_config)


def build_parquet_data(config: JobConfig, run_dir: Path, device: str) -> TrainDataBundle:
    data_dir = run_dir / "data"
    snapshot_dir = data_dir / "snapshot"
    plan_dir = data_dir / "plan"
    snapshot_manifest_path = snapshot_dir / "manifest.json"
    plan_manifest_path = plan_dir / "plan.json"

    if not snapshot_manifest_path.exists():
        build_snapshot_from_parquet(
            ParquetSnapshotConfig(
                input_paths=(config.run.dataset,),
                output_dir=str(snapshot_dir),
                snapshot_id=f"{run_dir.name}-snapshot",
                dataset_name=Path(config.run.dataset).stem,
                split="train",
                text_column=config.data.text_column,
                tokenizer_name=config.data.tokenizer_name,
                max_tokens_per_shard=config.data.max_tokens_per_shard,
                max_documents=config.data.max_documents,
            )
        )

    if not plan_manifest_path.exists():
        build_sequence_plan(
            SequencePlanSpec(
                snapshot_path=str(snapshot_dir),
                output_dir=str(plan_dir),
                sequence_length=config.model.context_length,
                stride=config.data.plan_stride,
                shuffle=config.data.shuffle,
                shuffle_seed=config.data.shuffle_seed,
            )
        )

    train_dataset = CausalLMSequenceDataset(str(snapshot_dir), str(plan_dir))
    train_loader = DeterministicBatchLoader(
        train_dataset,
        global_batch_size=config.training.batch_size,
        device=device,
        drop_last=True,
        wrap=True,
    )
    return TrainDataBundle(
        train_loader=train_loader,
        dataset_size=len(train_dataset),
        vocab_size=train_dataset.get_vocab_size(),
        exact_resume=True,
    )


def build_default_data(config: JobConfig, run_dir: Path, device: str) -> TrainDataBundle:
    return build_parquet_data(config, run_dir, device)


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


DEFAULT_TRAIN_SPEC = TrainSpec(
    name="causal_lm",
    build_model=build_default_model,
    build_data=build_default_data,
    loss_fn=causal_lm_loss,
)
