# Typical Experiment Flow

`mirrorshift` expects preprocessed `mirrordata` artifacts at training time. The normal loop is:

1. preprocess parquet into a snapshot
2. build a plan for the target context length
3. verify the artifacts
4. train against `snapshot_path + plan_path`
5. resume from checkpoints when needed

## 1. Build A Snapshot

Input data is parquet with a `text` column.

```bash
uv run mirrordata prep-parquet /path/to/data/*.parquet \
  --output-dir /tmp/my_snapshot \
  --snapshot-id my-dataset \
  --dataset-name my-dataset \
  --split train \
  --text-column text \
  --tokenizer-name p50k_base
```

This writes:

```text
/tmp/my_snapshot/
  manifest.json
  shards/
    train-tokens-00000.bin
    train-tokens-00000.idx
    ...
```

## 2. Build A Plan

Plans are sequence-length-specific views over a snapshot.

```bash
uv run mirrordata build-plan \
  --snapshot-path /tmp/my_snapshot \
  --output-dir /tmp/my_plan_ctx4096 \
  --sequence-length 4096 \
  --shuffle-seed 0
```

This writes:

```text
/tmp/my_plan_ctx4096/
  plan.json
  sample_starts.npy
  sample_order.npy
```

## 3. Verify Artifacts

```bash
uv run mirrordata verify \
  --snapshot-path /tmp/my_snapshot \
  --plan-path /tmp/my_plan_ctx4096
```

Useful inspection commands:

```bash
uv run mirrordata info /tmp/my_snapshot/manifest.json
jq .total_tokens /tmp/my_snapshot/manifest.json
```

## 4. Run A Basic Training Job

```bash
uv run mirrorshift-train \
  --job.config_file mirrorshift/config/train_configs/small.toml \
  --training.device cuda \
  --data.snapshot_path /tmp/my_snapshot \
  --data.plan_path /tmp/my_plan_ctx4096 \
  --model.context_length 4096 \
  --run.wandb_mode online \
  --training.max_steps 1000 \
  --training.log_every 1
```

Important details:

- `mirrorshift` trains against `snapshot_path` and `plan_path`, not raw parquet.
- `model.context_length` must match the plan sequence length.
- `training.batch_size` is the global batch size.
- run metrics go to W&B unless `run.wandb_mode=disabled`.

## 5. Enable Checkpointing / Resume

First run:

```bash
uv run mirrorshift-train \
  --job.config_file mirrorshift/config/train_configs/small.toml \
  --training.device cuda \
  --data.snapshot_path /tmp/my_snapshot \
  --data.plan_path /tmp/my_plan_ctx4096 \
  --model.context_length 4096 \
  --run.id my-run \
  --checkpoint.enable true \
  --checkpoint.interval 100 \
  --checkpoint.keep_latest_k 2 \
  --training.max_steps 500
```

Resume the latest checkpoint:

```bash
uv run mirrorshift-train \
  --job.config_file mirrorshift/config/train_configs/small.toml \
  --training.device cuda \
  --data.snapshot_path /tmp/my_snapshot \
  --data.plan_path /tmp/my_plan_ctx4096 \
  --model.context_length 4096 \
  --run.id my-run \
  --checkpoint.enable true \
  --checkpoint.interval 100 \
  --checkpoint.keep_latest_k 2 \
  --checkpoint.load_step -1 \
  --training.max_steps 1000
```

Resume notes:

- `run.id` must be fixed when resuming.
- Keep model/data/parallelism settings the same across resume legs.
- `checkpoint.keep_latest_k=0` disables pruning.

## 6. Common Experiment Toggles

Compile:

```bash
--compile.enable
```

Activation checkpointing:

```bash
--activation_checkpoint.mode full
--activation_checkpoint.mode selective \
--activation_checkpoint.selective_ac_option 2
--activation_checkpoint.mode selective \
--activation_checkpoint.selective_ac_option op
```

Deterministic runs:

```bash
--debug.seed 1234 \
--debug.deterministic
```

Optional gradient clipping:

```bash
--training.max_grad_norm 1.0
```

## 7. Distributed Launch

Use `torchrun` for multi-GPU jobs.

Replicated DP:

```bash
uv run torchrun --standalone --nproc_per_node 2 -m mirrorshift.train \
  --job.config_file mirrorshift/config/train_configs/small.toml \
  --training.device cuda \
  --data.snapshot_path /tmp/my_snapshot \
  --data.plan_path /tmp/my_plan_ctx4096 \
  --model.context_length 4096 \
  --parallelism.dp_replicate 2 \
  --parallelism.dp_shard 1
```

Sharded DP:

```bash
uv run torchrun --standalone --nproc_per_node 2 -m mirrorshift.train \
  --job.config_file mirrorshift/config/train_configs/small.toml \
  --training.device cuda \
  --data.snapshot_path /tmp/my_snapshot \
  --data.plan_path /tmp/my_plan_ctx4096 \
  --model.context_length 4096 \
  --parallelism.dp_replicate 1 \
  --parallelism.dp_shard 2
```

The required invariant is:

```text
WORLD_SIZE == dp_replicate * dp_shard
```

## 8. Run Outputs

Each run creates:

```text
runs/<run_id>/
  config.json
  manifest.json
  checkpoints/
    step-...
```

- `config.json` is the resolved config snapshot.
- `manifest.json` records the run metadata and input artifact paths.
- `checkpoints/` stores DCP checkpoints and checkpoint metadata.
