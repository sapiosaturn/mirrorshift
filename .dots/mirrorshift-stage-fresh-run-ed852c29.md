---
title: Stage fresh run artifacts behind atomic rename
status: open
priority: 1
issue-type: task
created-at: "2026-03-18T20:26:21.785212-07:00"
---

Done when fresh runs use a temporary staging directory that is atomically renamed into place after startup validation, retries after early failures no longer wedge a fixed run id, completed runs remain immutable, tests cover the lifecycle, and the changes are committed.
