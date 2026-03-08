---
title: Wire trainer to generic infra path
status: closed
priority: 1
issue-type: task
created-at: "\"2026-03-07T18:28:22.848800-08:00\""
closed-at: "2026-03-07T18:42:08.396856-08:00"
close-reason: Trainer, data loading, metrics, run artifacts, and checkpoint metadata are wired through the new runtime context and infra path.
---

Done when training uses the new infra layer for activation checkpointing and compile, and conditionally applies data parallel wrapping when configured.
