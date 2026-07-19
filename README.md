# Tennis Analyzer

Tennis Analyzer is a single-server FastAPI application that accepts untrusted tennis videos, queues long-running computer-vision analysis in RQ, persists status in PostgreSQL, and publishes a browser-playable annotated MP4 plus structured JSON. The same typed pipeline powers the worker and local CLI.

> Screenshot placeholder: upload and completed-job screenshots should be added after branding and deployment hostname are finalized.

## Current capabilities and limitations

The application streams uploads, validates them with FFprobe, and runs one complete background analysis. Frame-aligned results are saved as a reusable artifact. Users can then create multiple render variants with different overlays without loading inference models again. For fixed-camera video, users can also correct four outer court corners on a representative frame; future renders use that saved static calibration without model inference. Jobs report stage progress, allow cooperative cancellation, range-stream outputs, preserve audio where practical, expose safe failures, and delete database/file state together.

Event detection, in/out calls, speeds, distances, point splitting, and player roles are experimental monocular-video estimates. Model provenance is unknown and weights are not distributed. Analysis uses bounded low-memory passes: each selected model is loaded once per job stage, TrackNet preserves two source frames across chunk boundaries, continuity filtering runs over the complete coordinate track, and rendering streams directly to the encoder. Cancellation cannot interrupt a model call already in progress. Automatic retention and stale-job reconciliation are documented but not scheduled. There is no authentication: UUID detail links are unguessable, but the recent-jobs page lists this deployment's jobs. Use Nginx Proxy Manager access control for any non-private deployment.

## Architecture and layout

FastAPI/Jinja2 and a small local stylesheet form the web tier. PostgreSQL stores job metadata, Redis/RQ runs one ML job at a time, OpenCV performs bounded frame inference/rendering, and FFmpeg probes and normalizes H.264/AAC MP4 output. See [architecture](docs/ARCHITECTURE.md), [performance and constrained-host operation](docs/PERFORMANCE.md), [audit](docs/REPOSITORY_AUDIT.md), [options](docs/ANALYSIS_OPTIONS.md), and [result schema](docs/RESULT_SCHEMA.md).

```text
app/                    web routes, settings, DB model, services, templates, worker
alembic/                database migrations
tennis_analyzer/        typed pipeline, video safety, result/options schema, CLI
tests/                  generated-video, API, storage, state, pipeline, worker tests
docs/                   audit, architecture, deployment, options, schema
scripts/                model installation/checksum helper
models/                 model manifest only; weights ignored
deploy/                 self-contained pull-only VPS deployment bundle
.github/workflows/      tests, image build, Trivy scan, GHCR publishing
data/jobs/<uuid>/       runtime uploads and outputs; ignored
BallTrack/              retained third-party-derived git submodule
*.py                    legacy CV algorithms used through adapters
```

## Models

Initialize the submodule and install legally obtained local weights:

```bash
git submodule update --init --recursive
python scripts/download_models.py --source weights --destination models
```

The exact filenames, observed local checksums/sizes, feature mapping, and unresolved licensing are in [models/README.md](models/README.md). There are deliberately no invented download URLs. Model-free scene-cut detection and frame numbering can be tested without weights. At application startup/import models are not loaded; a selected worker stage fails clearly if its required file is missing.

## Local development

Requires Python 3.11, FFmpeg/FFprobe, PostgreSQL, Redis, and roughly 8 GB RAM for CPU model use.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[ml,dev]"
cp .env.example .env                     # Windows: copy .env.example .env
# For local SQLite development, set DATABASE_URL=sqlite:///./data/tennis.db
alembic upgrade head
uvicorn app.main:app --reload
```

In another terminal:

```bash
rq worker --url redis://localhost:6379/0 analysis
```

The site is at `http://localhost:8000`. The API creates a job with multipart fields:

```bash
curl -i -F "video=@sample.mp4;type=video/mp4" \
  -F "scene_cut_detection=true" -F "frame_number=true" \
  http://localhost:8000/api/jobs
curl http://localhost:8000/api/jobs/JOB_UUID
```

The CLI calls the identical pipeline:

```bash
tennis-analyze sample.mp4 data/cli-job --models models
tennis-analyze sample.mp4 data/cli-full --models models --full --device cpu
```

## CI/CD and container images

[`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml) runs on pull requests to `main`, pushes to `main`, `v*.*.*` tags, and manual dispatch. It runs formatting, linting, and tests; builds the production Dockerfile with BuildKit caching; scans the image with Trivy and uploads non-blocking SARIF findings when GitHub code scanning supports it; and publishes non-PR builds to `ghcr.io/mbonatte/tennis`. Review and prioritize code-scanning alerts separately; the presence of an upstream base-image advisory does not prevent publishing an otherwise verified image.

Pull requests build and scan without publishing. A successful `main` push publishes `latest` and `sha-<short-sha>`. A tag such as `v1.2.3` publishes `1.2.3`, `1.2`, `1`, and its SHA tag. The workflow uses `GITHUB_TOKEN`, needs no custom repository secret for publishing, and grants `security-events:write` only to the scan job and `packages:write` only to the publish job; all jobs otherwise use `contents:read`. GHCR package visibility is managed separately in GitHub. Private-package VPS pulls require a PAT with `read:packages`; never store it in `.env`.

Run CI checks locally:

```bash
python -m pip install -e ".[dev]"
python -m ruff format --check app tennis_analyzer tests ball.py
python -m ruff check app tennis_analyzer tests ball.py
python -m pytest -q
```

Build the production image locally for testing:

```bash
git submodule update --init --recursive
docker build -t tennis:local .
```

## Docker development

The default image is CPU-only and containers run as UID 10001. Compose defines `web`, `worker`, `postgres`, `redis`, and a one-shot `migrate`; named volumes hold DB, Redis, and job data, while the ignored local `./models` directory is mounted read-only. The worker count defaults to one.

```bash
cp .env.example .env
# Edit every placeholder. The application safely builds DATABASE_URL from
# the single POSTGRES_PASSWORD value.
docker network create proxy              # once; reuse your NPM external network
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f web worker
```

Apply later migrations exactly once:

```bash
docker compose run --rm migrate alembic upgrade head
```

This root Compose file is for local development and intentionally uses `build:`. Nginx Proxy Manager can proxy to `tennis-web:8000` on `proxy`.

## VPS deployment from the published image

Copy only the [`deploy/`](deploy/) directory to the VPS. Its Compose file has no `build:` and uses the same configurable `APP_IMAGE` for web, worker, and migrations:

```bash
cd deploy
cp .env.example .env
# Edit .env and place legally obtained weights in ./models
docker network create proxy 2>/dev/null || true
docker compose pull
docker compose run --rm migrate
docker compose up -d
docker compose ps
docker compose logs -f web
docker compose logs -f worker
```

Update with the same `pull`, explicit migration, and `up -d` sequence. Stop without deleting data using `docker compose down`. Do not casually use `docker compose down -v`, which deletes named database, Redis, and job-data volumes. Pin `APP_IMAGE` to `sha-<short-sha>` or a release tag for deterministic rollback; `latest` is convenient but mutable. Full GHCR login, persistence, backup, upgrade, rollback, Nginx Proxy Manager, and troubleshooting guidance is in [deploy/README.md](deploy/README.md) and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

For optional NVIDIA deployment, install NVIDIA Container Toolkit, change the image/build to a CUDA-compatible PyTorch base, reserve the GPU in an override, and set `DEVICE=cuda`. CUDA is intentionally not required or enabled in the default build.

## Configuration

All settings are environment validated. Important variables are `ENVIRONMENT`, `SECRET_KEY`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `REDIS_URL`, `DATA_ROOT`, `MODEL_ROOT`, `MAX_UPLOAD_BYTES`, `MAX_VIDEO_DURATION_SECONDS`, `MAX_VIDEO_WIDTH`, `MAX_VIDEO_HEIGHT`, `ALLOWED_VIDEO_EXTENSIONS`, `ALLOWED_VIDEO_CODECS`, `JOB_TIMEOUT_SECONDS`, `WORKER_CONCURRENCY`, `RETENTION_DAYS`, `LOG_LEVEL`, `DEVICE`, `ALLOWED_HOSTS`, `PUBLIC_BASE_URL`, `ANALYSIS_EXECUTION_MODE`, `ANALYSIS_CHUNK_FRAMES`, and `ANALYSIS_BALL_BATCH_SIZE`. Production builds `DATABASE_URL` safely from the single PostgreSQL password; an explicit `DATABASE_URL` remains supported for development/advanced deployments and is rejected when its password conflicts with `POSTGRES_PASSWORD`. Safe development defaults use local SQLite/data paths. Never commit `.env`.

## Tests and quality

Tests generate a tiny H.264/AAC video with FFmpeg and never download ML weights:

```bash
python -m pytest -q
python -m ruff check app tennis_analyzer tests ball.py
python -m ruff format --check app tennis_analyzer tests ball.py
```

An opt-in real-model smoke path is the CLI `--full` command above; keep its input very short. GitHub Actions performs the clean-checkout Linux image build and security scan.

## Operations and troubleshooting

- `GET /healthz` checks the web process; `GET /readyz` checks PostgreSQL and Redis.
- If an upload is rejected, inspect its real codec/duration with `ffprobe input.mp4`; extensions and browser MIME are not trusted.
- If a job reports a missing model, compare `MODEL_ROOT` and the model-volume contents with `models/README.md`.
- If output is absent, inspect `docker compose logs worker` and RQ's failed registry. Public responses never include tracebacks.
- A killed worker may leave a DB row `running`; after `JOB_TIMEOUT_SECONDS`, confirm its RQ job is absent and mark it failed before retrying.
- Back up PostgreSQL and `job-data`; Redis AOF is persisted but is not a substitute for database/job-data backups.
- `RETENTION_DAYS` records policy only today. Delete through the UI or `DELETE /api/jobs/<uuid>` until scheduled cleanup is added.
- Before updating: stop/drain the worker, back up, select a tested immutable GHCR tag, pull it, run the one-shot migration, and recreate services.

The unauthenticated MVP is appropriate only behind a trusted boundary/access list. It lacks users, authorization, quotas, malware scanning, CSRF tokens, rate limiting, and per-tenant storage isolation. Authentication and ownership are the first production hardening priority.
