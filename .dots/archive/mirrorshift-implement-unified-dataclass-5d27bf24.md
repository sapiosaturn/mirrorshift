---
title: Implement unified dataclass+TOML+CLI config stack (torchtitan-lite)
status: closed
priority: 1
issue-type: task
created-at: "\"\\\"2026-02-15T18:44:54.380247-08:00\\\"\""
closed-at: "2026-02-15T19:05:14.498121-08:00"
close-reason: Implemented unified JobConfig dataclass package with TOML+CLI parsing (CLI > TOML > defaults), strict unknown-field checks, default TOML preset, train entrypoint migration, and passing tests/smoke run.
---

Done when mirrorshift uses one JobConfig dataclass tree loaded from TOML with dotted CLI overrides, strict unknown-field rejection, and resolved-config logging/snapshot support.
