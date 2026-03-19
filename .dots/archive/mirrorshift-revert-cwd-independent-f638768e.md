---
title: Revert cwd-independent path resolution hardening
status: closed
priority: 1
issue-type: task
created-at: "\"\\\"2026-03-18T20:52:19.253022-07:00\\\"\""
closed-at: "2026-03-18T20:55:17.534851-07:00"
close-reason: Reverted cwd-independent config/path normalization in ConfigManager, kept fake-data alias support, and updated tests after full pytest passed.
---

Done when ConfigManager no longer rewrites config/data/run paths for cwd-independent usage, tests match the simpler repo-root assumption, other hardening remains intact, and the changes are committed.
