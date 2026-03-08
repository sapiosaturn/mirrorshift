---
title: Make mirrorshift parquet-only for training input
status: closed
priority: 1
issue-type: task
created-at: "\"\\\"2026-03-07T16:10:08.292229-08:00\\\"\""
closed-at: "2026-03-07T16:15:15.529909-08:00"
close-reason: mirrorshift training and default config now assume parquet input only and use a tracked sample parquet dataset
---

Done when mirrorshift config/spec/trainer no longer support the old text input path and tests/default configs are updated for parquet-only data handling.
