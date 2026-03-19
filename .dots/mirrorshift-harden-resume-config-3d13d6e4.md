---
title: Harden resume config checks and path resolution
status: open
priority: 1
issue-type: task
created-at: "2026-03-18T20:18:55.465505-07:00"
---

Done when resume rejects config drift except for an explicit allowlist, default and TOML-relative paths no longer depend on caller cwd, tests cover both behaviors, and the changes are committed.
