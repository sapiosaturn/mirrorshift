---
title: Add simplified parallel/perf infra from torchtitan
status: closed
priority: 1
issue-type: task
created-at: "\"\\\"2026-03-07T18:26:16.132348-08:00\\\"\""
closed-at: "2026-03-07T18:42:08.401111-08:00"
close-reason: Added a simplified TorchTitan-style infra layer with generic DP shard/replicate, activation checkpointing, compile wiring, and passing coverage.
---

Done when mirrorshift has a model-agnostic infra layer for dp_replicate + dp_shard, activation checkpointing, and torch.compile setup derived from TorchTitan's current composable APIs, with tests and docs for CPU-safe development.
