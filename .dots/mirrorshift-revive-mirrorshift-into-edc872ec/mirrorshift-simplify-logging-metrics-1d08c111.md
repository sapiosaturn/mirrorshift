---
title: Simplify logging metrics and report reference metric sets
status: closed
priority: 1
issue-type: task
created-at: "\"\\\"2026-03-08T16:00:54.339633-07:00\\\"\""
closed-at: "2026-03-08T16:04:51.806318-07:00"
close-reason: Trimmed current training logging to train loss only and captured the reference metric sets from nmoe and TorchTitan for follow-up selection.
---

Remove perplexity from mirrorshift logging, keep only train loss in current metrics output, summarize the metric sets tracked by nmoe and TorchTitan, and commit the code cleanup.
