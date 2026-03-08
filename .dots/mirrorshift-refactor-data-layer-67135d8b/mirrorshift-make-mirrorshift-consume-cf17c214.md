---
title: Make mirrorshift consume prebuilt mirrordata artifacts
status: closed
priority: 1
issue-type: task
created-at: "\"\\\"2026-03-07T17:02:11.896579-08:00\\\"\""
closed-at: "2026-03-07T17:06:14.705687-08:00"
close-reason: mirrorshift now opens prebuilt snapshot and plan inputs directly and no longer preprocesses parquet during training
---

Done when mirrorshift config/spec/trainer open an existing snapshot and plan instead of preprocessing parquet during training.
