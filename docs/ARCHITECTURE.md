# Architecture

One FastAPI application serves Jinja pages and JSON endpoints. PostgreSQL is the source of truth for job state; Redis/RQ carries only execution messages. One RQ worker calls the same `tennis_analyzer.pipeline.analyze_video()` interface as the CLI. Web and worker share a job-data volume and the read-only `./models` bind mount.

Uploads stream into `data/jobs/<uuid>/input/source.<ext>`. FFprobe validates content, codec, duration, frame rate, and resolution before a full-analysis job is queued. The worker runs each model as a separate bounded pass and atomically saves a versioned `analysis-artifact.json` plus `result.json`; it does not render video. Each later `RenderOutput` streams the source with selected saved overlays, normalizes H.264/AAC output, and never loads an analysis model. One analysis can own many render variants. Routes resolve only database-owned relative paths beneath `DATA_ROOT`.

`ANALYSIS_EXECUTION_MODE=low_memory` is the supported production contract. The persistent all-model single-pass design exists only as a benchmark comparator because simultaneous model residency is unsafe on the initial 8 GB target. See [performance and constrained-host operation](PERFORMANCE.md).

Job transitions are explicit: `uploaded -> queued -> running -> completed|failed|cancelled`; queued jobs may cancel directly. A worker exception is caught at the application boundary, with a safe message in the public field and traceback only in `internal_diagnostic`.

RQ's job timeout covers hard hangs. Cooperative cancellation is checked between chunks/stages. If a worker dies, RQ moves the queue job to its failed registry, but the database row can remain `running`; operators should run a periodic reconciliation that marks rows stale when their RQ job is absent and `started_at` exceeds the timeout. This reconciliation and retention scheduler are documented operational steps, not yet automated.

The UUID is both identifier and unguessable access token for this unauthenticated MVP. It prevents sequential enumeration of detail routes, but `/jobs` is intentionally a server-wide operational list. Put the site behind Nginx Proxy Manager access control or a private network until real authentication is added.
