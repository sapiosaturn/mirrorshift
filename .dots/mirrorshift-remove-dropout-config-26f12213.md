---
title: Remove dropout configuration and dead dropout code paths
status: closed
priority: 1
issue-type: task
created-at: "\"2026-02-15T17:48:42.820639-08:00\""
closed-at: "2026-02-15T18:00:13.742642-08:00"
close-reason: Removed dropout keys from model config schema and defaults; model/trainer now run with explicit zero-dropout behavior.
blocks:
  - mirrorshift-add-checkpoint-manager-3c7130f8
---

Done when dropout fields/calls are removed from configs and model/training code consistently uses zero-dropout behavior.
