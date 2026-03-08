---
title: Add simplified parallel/perf infra from torchtitan
status: active
priority: 1
issue-type: task
created-at: "\"2026-03-07T18:26:16.132348-08:00\""
---

Done when mirrorshift has a model-agnostic infra layer for dp_replicate + dp_shard, activation checkpointing, and torch.compile setup derived from TorchTitan's current composable APIs, with tests and docs for CPU-safe development.
