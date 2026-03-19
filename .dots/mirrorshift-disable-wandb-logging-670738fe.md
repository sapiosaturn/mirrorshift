---
title: Disable wandb logging in test runs
status: closed
priority: 1
issue-type: task
created-at: "\"\\\"2026-02-15T19:35:44.358097-08:00\\\"\""
closed-at: "2026-02-15T19:36:29.017251-08:00"
close-reason: Added metrics logger factory that returns NoOpLogger when PYTEST_CURRENT_TEST is present or wandb_mode=disabled; train now uses factory and tests cover no-op behavior.
---

Done when pytest runs always use a no-op metrics logger so no wandb init/log calls occur in tests.
