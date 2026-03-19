---
title: Rewire mirrorshift to depend on mirrordata
status: closed
priority: 2
issue-type: task
created-at: "\"\\\"2026-03-07T15:17:13.439718-08:00\\\"\""
closed-at: "2026-03-07T15:22:17.850730-08:00"
close-reason: Wired mirrorshift to depend on the local mirrordata workspace package, rewrote dataset imports, and removed the legacy mirrorshift/data.py module.
---

Done when mirrorshift pyproject depends on the local mirrordata package, legacy mirrorshift/data.py is removed, and current imports build against mirrordata.
