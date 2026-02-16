---
title: Simplify trainer by removing sampling and inference module
status: closed
priority: 1
issue-type: task
created-at: "\"\\\"2026-02-15T18:22:50.296391-08:00\\\"\""
closed-at: "2026-02-15T18:24:13.577740-08:00"
close-reason: Removed sampling from trainer/config, deleted inference module and tests, cleaned package/docs references, and verified with pytest.
---

Done when sampling hooks are removed from trainer/config, inference.py is deleted, and tests/docs are updated and passing.
