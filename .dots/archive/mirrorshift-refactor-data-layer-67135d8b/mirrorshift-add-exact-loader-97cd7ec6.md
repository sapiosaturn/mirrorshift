---
title: Add exact loader-state resume for mirrordata training
status: closed
priority: 1
issue-type: task
created-at: "\"\\\"2026-03-07T17:25:08.552644-08:00\\\"\""
closed-at: "2026-03-07T17:33:10.214367-08:00"
close-reason: checkpoints now persist deterministic loader cursor state and mirrorshift resumes from it exactly when available
---

Done when checkpoints persist loader cursor state and mirrorshift resumes from it instead of reconstructing position from step*batch_size.
