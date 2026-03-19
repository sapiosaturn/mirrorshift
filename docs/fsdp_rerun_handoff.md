# Mirrorshift FSDP Rerun Handoff

This file is for the next Codex instance running on the CUDA machine.

## Goal

Re-run the previously failing `dp_shard > 1` validation after the grad-norm fix.

Everything below is intentionally narrow. Single-GPU CUDA, checkpoint/resume, determinism, compile, activation checkpointing, and DDP-style replicate were already validated. The remaining blocker was FSDP-style sharding crashing in gradient norm handling.

## What Changed

- `mirrorshift` now uses a TorchTitan-style DTensor-aware grad norm path.
- Gradient norm retrieval no longer manually multiplies DTensor grads.
- Optional grad clipping support was added via `training.max_grad_norm`, but the default remains `null`, so training behavior is unchanged unless explicitly enabled.

## Environment / Known Paths

```bash
cd /home/ubuntu/ml_repos/mirrorshift

export REPO=/home/ubuntu/ml_repos/mirrorshift
export LOG_DIR=/home/ubuntu/mirrorshift-runs
export SNAPSHOT=/home/ubuntu/datasets/dclm10b-snapshot
export PLAN=/home/ubuntu/datasets/dclm10b-plan-ctx4096
export CONFIG=$REPO/mirrorshift/config/train_configs/small.toml
export WANDB_PROJECT=mirrorshift-gpu-validation

COMMON_ARGS=(
  --job.config_file "$CONFIG"
  --run.log_dir "$LOG_DIR"
  --run.wandb_project "$WANDB_PROJECT"
  --run.wandb_mode online
  --data.snapshot_path "$SNAPSHOT"
  --data.plan_path "$PLAN"
  --training.device cuda
  --model.context_length 4096
  --training.batch_size 16
  --training.log_every 1
  --checkpoint.enable true
  --checkpoint.interval 50
  --checkpoint.keep_latest_k 2
  --parallelism.dp_replicate 1
  --parallelism.dp_shard 1
  --compile.enable false
  --activation_checkpoint.mode none
)
```

`training.batch_size` is still the global batch size.

## Required Runs

### R1. FSDP Baseline on 2 GPUs

```bash
uv run torchrun --standalone --nproc_per_node 2 -m mirrorshift.train \
  "${COMMON_ARGS[@]}" \
  --run.id gpu-fsdp-2 \
  --debug.seed 1234 \
  --debug.deterministic false \
  --training.max_steps 120 \
  --parallelism.dp_replicate 1 \
  --parallelism.dp_shard 2
```

Success criteria:
- no DTensor assertion in grad norm
- training reaches step 120
- checkpoints at 100 and 120
- `train/grad_norm` is finite and logged normally

### R2. FSDP Resume Validation on 2 GPUs

Leg A:

```bash
uv run torchrun --standalone --nproc_per_node 2 -m mirrorshift.train \
  "${COMMON_ARGS[@]}" \
  --run.id gpu-fsdp-resume \
  --debug.seed 1234 \
  --debug.deterministic false \
  --training.max_steps 100 \
  --parallelism.dp_replicate 1 \
  --parallelism.dp_shard 2
```

Leg B:

```bash
uv run torchrun --standalone --nproc_per_node 2 -m mirrorshift.train \
  "${COMMON_ARGS[@]}" \
  --run.id gpu-fsdp-resume \
  --debug.seed 1234 \
  --debug.deterministic false \
  --training.max_steps 200 \
  --parallelism.dp_replicate 1 \
  --parallelism.dp_shard 2 \
  --checkpoint.load_step -1
```

Success criteria:
- Leg A writes checkpoints
- Leg B resumes from step 100
- LR and `n_tokens_seen` continue instead of resetting
- latest two checkpoints remain after completion

## Optional Follow-Up Runs

Only do these if `R1` and `R2` both pass cleanly.

### R3. HSDP-Style 2x2 Mesh on 4 GPUs

```bash
uv run torchrun --standalone --nproc_per_node 4 -m mirrorshift.train \
  "${COMMON_ARGS[@]}" \
  --run.id gpu-hsdp-2x2 \
  --debug.seed 1234 \
  --debug.deterministic false \
  --training.max_steps 120 \
  --parallelism.dp_replicate 2 \
  --parallelism.dp_shard 2
```

### R4. Explicit Grad Clipping Smoke

This is only to prove the new config path works.

```bash
uv run mirrorshift-train "${COMMON_ARGS[@]}" \
  --run.id gpu-max-grad-norm \
  --debug.seed 1234 \
  --debug.deterministic false \
  --training.max_steps 60 \
  --training.max_grad_norm 1.0
```

## What To Look At

### 1. Immediate Fix Validation

- The old DTensor assertion in grad norm should be gone.
- `train/grad_norm` should log normally on FSDP runs.
- The first logged step should look healthy instead of crashing before metrics.

### 2. Resume Continuity

For `gpu-fsdp-resume`:
- resume should start from step 100
- `optimizer/lr` should continue the schedule
- `train/n_tokens_seen` should continue monotonically

With `batch_size=16` and `context_length=4096`, `train/n_tokens_seen` should still increase by:

```text
65536 tokens per step
```

### 3. Checkpoint Integrity

Inspect the run dir:
- `config.json`
- `manifest.json`
- `checkpoints/step-*`

Inspect `checkpoint_meta.json` and confirm `data_identity` matches the expected snapshot/plan fingerprints.

### 4. Distributed Health

- no NCCL hangs
- no rank-specific failures
- no NaN/inf in loss or grad norm

## Expected Outcome

If `R1` and `R2` pass, the main remaining blocker for calling the repo properly revived is gone. At that point, the core single-GPU path, checkpoint/resume path, compile path, activation checkpointing path, DDP path, and FSDP path have all been exercised on real CUDA hardware.
