# Mirrorshift GPU Validation Handoff

This file is for the next Codex instance running on the CUDA machine.

## Goal

Exercise the current `mirrorshift` training stack enough to answer:

1. Does single-GPU CUDA training work reliably?
2. Do checkpoint save/load and exact data resume work?
3. Do `torch.compile` and activation checkpointing work on the real target environment?
4. Do the new composable distributed paths work:
   - `dp_replicate > 1, dp_shard = 1` (DDP-style replicate)
   - `dp_replicate = 1, dp_shard > 1` (FSDP-style shard)
   - optionally `dp_replicate > 1, dp_shard > 1` if enough GPUs are available

## Known Context

- Dataset snapshot already exists:
  - `/home/ubuntu/datasets/dclm10b-snapshot`
- Context-4096 plan already exists:
  - `/home/ubuntu/datasets/dclm10b-plan-ctx4096`
- A single-GPU CUDA run has already been exercised manually with and without `--compile.enable`.
- `training.batch_size` is the **global** batch size, not per-rank.
  - Keep it fixed across single-GPU, DDP, and FSDP runs if the goal is apples-to-apples correctness comparison.
- Current W&B behavior:
  - `mirrorshift` uses `wandb.init(name=run_id, ...)`, not an explicit W&B run id/resume policy.
  - Resume legs will likely show up as a second W&B run with the same display name.
  - Use local run artifacts plus console logs as the source of truth for checkpoint/resume validation.
- Current resume limitation:
  - `mirrorshift` validates snapshot/plan identity on resume, but it does **not** yet reject arbitrary config drift.
  - For resume tests, keep all training/model/data/parallelism flags identical except:
    - `--training.max_steps`
    - `--checkpoint.load_step`

## Preflight

Run these first and record the output in the experiment notes:

```bash
cd /home/ubuntu/ml_repos/mirrorshift

pwd
git rev-parse HEAD
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

## Shared Args

Use absolute paths everywhere.

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

If any run OOMs, lower `--training.batch_size` first and keep the reduced value fixed across the comparable runs.

## Must-Run Experiments

### E1. Single-GPU Baseline + Checkpoint

Why:
- Establish the clean CUDA baseline.
- Validate metrics, run artifacts, checkpoint save path, and basic learning behavior.

```bash
uv run mirrorshift-train "${COMMON_ARGS[@]}" \
  --run.id gpu-baseline \
  --debug.seed 1234 \
  --debug.deterministic true \
  --training.max_steps 150
```

Expected:
- No hangs, NaNs, or device errors.
- Checkpoints at steps `50`, `100`, `150`.
- Loss should trend downward or at least remain sane for the tiny model.

### E2. Checkpoint Resume Correctness

Why:
- This is the highest-value functional test after the baseline.
- Validates DCP save/load, optimizer state restore, and exact loader cursor restore.

Leg A:

```bash
uv run mirrorshift-train "${COMMON_ARGS[@]}" \
  --run.id gpu-resume \
  --debug.seed 1234 \
  --debug.deterministic true \
  --training.max_steps 120
```

Leg B:

```bash
uv run mirrorshift-train "${COMMON_ARGS[@]}" \
  --run.id gpu-resume \
  --debug.seed 1234 \
  --debug.deterministic true \
  --training.max_steps 240 \
  --checkpoint.load_step -1
```

Expected:
- Leg B logs `Resuming training from step=120`.
- `optimizer/lr` continues from step 121 rather than resetting.
- `train/n_tokens_seen` continues smoothly.
- Local run dir should still have only the latest two checkpoint directories because `keep_latest_k=2`.

### E3. Determinism Pair

Why:
- Verifies the new `seed` + deterministic path on a real CUDA machine.
- Best isolated with `compile=false` and `activation_checkpoint.mode=none`.

Run A:

```bash
uv run mirrorshift-train "${COMMON_ARGS[@]}" \
  --run.id gpu-determinism-a \
  --debug.seed 2025 \
  --debug.deterministic true \
  --training.max_steps 60
```

Run B:

```bash
uv run mirrorshift-train "${COMMON_ARGS[@]}" \
  --run.id gpu-determinism-b \
  --debug.seed 2025 \
  --debug.deterministic true \
  --training.max_steps 60
```

Expected:
- Stepwise loss and LR should match exactly or be extremely close.
- If they diverge early, note the first differing step and the metric that diverged first.

### E4. Single-GPU Compile

Why:
- `torch.compile` is one of the major new codepaths.
- Needs separate validation because compile warmup and graph capture change runtime behavior.

```bash
uv run mirrorshift-train "${COMMON_ARGS[@]}" \
  --run.id gpu-compile \
  --debug.seed 1234 \
  --debug.deterministic false \
  --training.max_steps 150 \
  --compile.enable true
```

Expected:
- No compile-time crash or graph-break-induced instability.
- Ignore the first several steps when evaluating throughput.

### E5. Single-GPU Activation Checkpointing: Full

Why:
- Exercises the simplest activation checkpoint wrapper path.
- Validates memory reduction behavior against baseline.

```bash
uv run mirrorshift-train "${COMMON_ARGS[@]}" \
  --run.id gpu-ac-full \
  --debug.seed 1234 \
  --debug.deterministic false \
  --training.max_steps 150 \
  --activation_checkpoint.mode full
```

Expected:
- Lower peak memory than the baseline.
- Some throughput regression is normal.

### E6. Single-GPU Activation Checkpointing: Selective Layer Mode

Why:
- Exercises the integer selective path, which is distinct from `full`.

```bash
uv run mirrorshift-train "${COMMON_ARGS[@]}" \
  --run.id gpu-ac-selective-2 \
  --debug.seed 1234 \
  --debug.deterministic false \
  --training.max_steps 150 \
  --activation_checkpoint.mode selective \
  --activation_checkpoint.selective_ac_option 2
```

### E7. Single-GPU Activation Checkpointing: Selective Op Mode

Why:
- Exercises the most specialized AC path and is therefore worth a direct shakedown.

```bash
uv run mirrorshift-train "${COMMON_ARGS[@]}" \
  --run.id gpu-ac-selective-op \
  --debug.seed 1234 \
  --debug.deterministic false \
  --training.max_steps 150 \
  --activation_checkpoint.mode selective \
  --activation_checkpoint.selective_ac_option op
```

### E8. Single-GPU Compile + Activation Checkpointing

Why:
- Verifies the combined order-of-operations path now used by `mirrorshift`:
  - activation checkpointing
  - compile
  - materialize/init

```bash
uv run mirrorshift-train "${COMMON_ARGS[@]}" \
  --run.id gpu-compile-ac \
  --debug.seed 1234 \
  --debug.deterministic false \
  --training.max_steps 150 \
  --compile.enable true \
  --activation_checkpoint.mode selective \
  --activation_checkpoint.selective_ac_option 2
```

## Distributed Experiments

Only run these if the machine has enough GPUs.

### E9. DDP-Style Replicate (`dp_replicate > 1`)

Why:
- Validates the composable `replicate(...)` path.

Use 2 GPUs first:

```bash
uv run torchrun --standalone --nproc_per_node 2 -m mirrorshift.train \
  "${COMMON_ARGS[@]}" \
  --run.id gpu-ddp-2 \
  --debug.seed 1234 \
  --debug.deterministic false \
  --training.max_steps 120 \
  --parallelism.dp_replicate 2 \
  --parallelism.dp_shard 1
```

If that passes cleanly and the box has 4+ GPUs, optionally repeat with `--nproc_per_node 4` and `--parallelism.dp_replicate 4`.

### E10. FSDP-Style Shard (`dp_shard > 1`)

Why:
- Validates the composable `fully_shard(...)` path and mixed distributed state handling.

Use 2 GPUs first:

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

If that passes cleanly and the box has 4+ GPUs, optionally repeat with `--nproc_per_node 4` and `--parallelism.dp_shard 4`.

### E11. FSDP Checkpoint + Resume

Why:
- Distributed checkpoint save/load with sharded state is an important proof point.

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

### E12. Optional 2D Mesh (`dp_replicate > 1` and `dp_shard > 1`)

Only run this if 4+ GPUs are available and E9/E10 already passed.

Why:
- Validates the combined 2D mesh path in `ParallelDims`.

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

## What To Look At After The Runs

### 1. Basic Health

- Any CUDA OOMs, NCCL hangs, collective mismatches, compile crashes, or checkpoint load failures.
- Any NaN/inf in:
  - `train/loss`
  - `train/max_loss`
  - `train/grad_norm`
- Whether the run exits cleanly and writes the manifest/config/checkpoints you expect.

### 2. Step Accounting

With:
- `training.batch_size = 16`
- `model.context_length = 4096`

the expected token increment is:

```text
16 * 4096 = 65536 tokens per step
```

Check that:
- `train/n_tokens_seen` increases by `65536` every step.
- On resume, it continues from the previous leg instead of resetting.

### 3. Resume Correctness

For `gpu-resume` and `gpu-fsdp-resume`:

- `optimizer/lr` at resumed step `N+1` should match the schedule continuation, not a warmup restart.
- `train/loss` after resume should be in the same neighborhood as the end of the previous leg.
- Local run dir should contain:
  - `config.json`
  - `manifest.json`
  - `checkpoints/step-*`
- With `keep_latest_k=2`, only the latest two checkpoint directories should remain.
- Inspect `checkpoints/step-*/checkpoint_meta.json` and confirm the saved `data_identity` matches the snapshot/plan in use.

### 4. Determinism

For `gpu-determinism-a` vs `gpu-determinism-b`:

- Compare the per-step curves for:
  - `train/loss`
  - `train/max_loss`
  - `train/grad_norm`
  - `optimizer/lr`
- Ideal result:
  - exact match
- Acceptable result:
  - numerically tiny drift only
- Bad result:
  - early qualitative divergence

If it is bad, capture:
- first differing step
- first differing metric
- whether the divergence is visible in console logs, W&B, or both

### 5. Compile

For `gpu-compile` and `gpu-compile-ac`:

- Ignore early warmup/compile steps when judging throughput.
- Compare against baseline after warmup:
  - `throughput/tokens_per_second_per_gpu`
  - `throughput/tflops`
- Look for:
  - graph compile failures
  - very unstable throughput
  - much worse-than-baseline performance after warmup

### 6. Activation Checkpointing

Compare `gpu-ac-full`, `gpu-ac-selective-2`, and `gpu-ac-selective-op` against the baseline:

- `memory/max_active_gib`
- `memory/max_reserved_gib`
- `throughput/tokens_per_second_per_gpu`

Expected:
- lower memory
- some throughput penalty

Unexpected:
- memory not improving at all
- throughput collapsing dramatically
- large loss instability relative to the baseline

### 7. Distributed Parity

For DDP/FSDP/HSDP runs:

- Loss scale should stay comparable to single-GPU runs when the global batch size is fixed.
- `train/n_tokens_seen` should still increase by `65536` per step.
- No rank-specific crashes or deadlocks.
- Primary-rank logs should look normal and monotonic.

### 8. Final Deliverable Back From The GPU Box

The next Codex instance should return:

1. A table of all attempted runs:
   - run id
   - config deltas from baseline
   - pass/fail
   - short failure reason if any
2. A short summary of what is solid now.
3. A short list of what still looks risky or broken.
4. The single next most valuable code fix, if any failure pattern is found.
