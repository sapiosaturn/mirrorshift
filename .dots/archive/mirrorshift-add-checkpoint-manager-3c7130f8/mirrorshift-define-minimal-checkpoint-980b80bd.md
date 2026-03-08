---
title: Define minimal checkpoint config and file layout
status: closed
priority: 1
issue-type: task
created-at: "\"2026-03-07T14:09:17.669337-08:00\""
closed-at: "2026-03-07T14:22:47.227317-08:00"
close-reason: "Defined and committed the minimal checkpoint surface in 6160bf5: enable, folder, interval, keep_latest_k, and explicit load_step-based resume semantics."
---

Done when mirrorshift has a deliberately minimal checkpoint config surface, on-disk layout, and documented non-goals for exact data-order resume, HF import/export, async save, and model-only loads.
