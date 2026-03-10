---
title: Add TorchTitan-style core training metrics
status: closed
priority: 1
issue-type: task
created-at: "\"\\\"2026-03-10T15:13:04.329701-07:00\\\"\""
closed-at: "2026-03-10T15:18:45.950552-07:00"
close-reason: Added TorchTitan-style core training metrics to the trainer and logger path, verified with full tests and CPU smoke.
---

Log global average loss, global max loss, grad norm, tokens/sec/gpu, tflops, end-to-end time, data-loading time, memory active/reserved GiB, lr, and n_tokens_seen in mirrorshift, with tests and commits.
