---
title: Move trainer to step-based API and remove epoch/validation flow
status: open
priority: 1
issue-type: task
created-at: "2026-02-15T17:48:42.813755-08:00"
blocks:
  - mirrorshift-introduce-trainspec-style-ad8b8103
---

Done when training is controlled by max_steps, epoch references are removed, and validation loop/config paths are deleted.
