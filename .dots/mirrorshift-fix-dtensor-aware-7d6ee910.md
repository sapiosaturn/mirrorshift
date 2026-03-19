---
title: Fix DTensor-aware grad norm handling for FSDP
status: active
priority: 1
issue-type: task
created-at: "\"2026-03-18T18:54:38.380683-07:00\""
---

Done when mirrorshift uses a TorchTitan-style grad norm path that works under dp_shard/FSDP, tests cover the behavior, and the GPU validation handoff doc is replaced with a focused rerun document for the next CUDA pass.
