---
title: Replace TensorBoard with wandb-only metrics
status: closed
priority: 1
issue-type: task
created-at: "\"\\\"2026-02-15T19:18:53.929101-08:00\\\"\""
closed-at: "2026-02-15T19:23:23.147230-08:00"
close-reason: Removed TensorBoard from training/tests/deps/docs, added minimal wandb-only logger with step-based metric logging, and verified with pytest + offline smoke run.
---

Done when TensorBoard is removed from code/deps/docs and training logs metrics only to wandb with minimal config fields.
