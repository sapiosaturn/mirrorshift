---
title: Fix mirrordata special-token encoding and add prep logging
status: closed
priority: 1
issue-type: task
created-at: "\"\\\"2026-03-16T23:18:07.535143-07:00\\\"\""
closed-at: "2026-03-16T23:20:26.388836-07:00"
close-reason: Mirrordata preprocessing now treats literal tiktoken special-token strings as normal text, emits useful progress logging, and is covered by tests.
---

Treat literal tiktoken special-token strings as normal text during parquet preprocessing, add useful preprocessing logging/progress output, add tests, verify, and commit the change.
