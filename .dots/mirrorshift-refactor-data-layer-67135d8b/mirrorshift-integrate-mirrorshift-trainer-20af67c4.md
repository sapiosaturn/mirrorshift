---
title: Integrate mirrorshift trainer with mirrordata runtime
status: closed
priority: 1
issue-type: task
created-at: "\"\\\"2026-03-07T15:49:29.863687-08:00\\\"\""
closed-at: "2026-03-07T16:01:24.598265-08:00"
close-reason: mirrorshift trainer now preprocesses parquet inputs into run-local mirrordata snapshots/plans and trains through the deterministic mirrordata loader
---

Done when mirrorshift training can preprocess parquet input via mirrordata, build a sequence plan in the run directory, and train using the mirrordata runtime loader.
