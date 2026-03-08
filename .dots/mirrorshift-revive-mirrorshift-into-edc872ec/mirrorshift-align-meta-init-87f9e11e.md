---
title: Align meta init ordering with TorchTitan
status: closed
priority: 1
issue-type: task
created-at: "\"\\\"2026-03-08T16:00:54.339628-07:00\\\"\""
closed-at: "2026-03-08T16:04:51.803838-07:00"
close-reason: "Moved model infra ahead of materialization/init so mirrorshift now follows TorchTitan's higher-level meta-model ordering, with tests and CPU smoke passing."
---

Apply model infra to the meta model before materialization/init, keep tests green, and commit the change.
