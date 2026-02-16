---
title: Move trainer to step-based API and remove epoch/validation flow
status: closed
priority: 1
issue-type: task
created-at: "\"2026-02-15T17:48:42.813755-08:00\""
closed-at: "2026-02-15T18:00:13.742636-08:00"
close-reason: Refactored trainer to max_steps-based loop, removed epoch/validation flow, and simplified logging/sampling triggers.
blocks:
  - mirrorshift-introduce-trainspec-style-ad8b8103
---

Done when training is controlled by max_steps, epoch references are removed, and validation loop/config paths are deleted.
