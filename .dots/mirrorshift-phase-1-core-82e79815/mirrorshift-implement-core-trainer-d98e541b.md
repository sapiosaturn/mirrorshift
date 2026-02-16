---
title: Implement core trainer/config refactor with CPU smoke tests and unit coverage
status: closed
priority: 1
issue-type: task
created-at: "\"\\\"2026-02-15T17:53:41.500098-08:00\\\"\""
closed-at: "2026-02-15T18:00:13.745324-08:00"
close-reason: Implemented core trainer/config refactor and added fast CPU unit smoke coverage for configs, modules, inference, experiments, and train loop.
blocks:
  - mirrorshift-introduce-trainspec-style-ad8b8103
---

Done when trainer is step-based and minimal, config paths are refactored for fail-fast behavior, and fast unit tests cover forward/backward, config validation, and core modules.
