# VPS deployment

This directory is the complete deployment bundle. The VPS does not need the Git repository, Dockerfile, Python, build tools, or application source. Copy `compose.yaml`, `.env.example`, this README, and the empty `models/` directory to the server. GitHub Actions builds the application once and publishes it to GHCR.

## Prerequisites

- Docker Engine with the Compose v2 plugin.
- The external Docker network used by Nginx Proxy Manager: `docker network create proxy` (run once if it does not exist).
- Enough disk for PostgreSQL, Redis AOF, uploads, outputs, and model weights.
- Access to `ghcr.io/mbonatte/tennis` if the package is private.

PostgreSQL and Redis are only attached to the internal `backend` network and publish no host ports. The web container joins `proxy` and exposes port 8123 only to containers on that network, avoiding a collision with an existing port-8000 container. Configure Nginx Proxy Manager to forward to `tennis-web:8123`, enable TLS and an access list, and set `client_max_body_size 2048m;` in its advanced configuration.

## First deployment

```bash
cd deploy
cp .env.example .env
mkdir -p models
chmod 700 models
# Edit .env: change every placeholder and set the real hostname/public URL.
# DATABASE_URL is built safely from the single POSTGRES_PASSWORD setting.

docker network create proxy 2>/dev/null || true
docker compose pull
docker compose run --rm migrate
docker compose up -d
```

Inspect the deployment:

```bash
docker compose ps
docker compose logs -f web
docker compose logs -f worker
docker compose logs postgres redis
curl -fsS https://your-host.example/healthz
curl -fsS https://your-host.example/readyz
```

The `migrate` service uses the `tools` profile so ordinary `docker compose up -d` does not run migrations from every replica. Explicit `docker compose run --rm migrate` is the only migration step.

## Image versions and GHCR access

`APP_IMAGE` selects the exact image. `latest` follows the newest successful push to `main`, but production should use an immutable commit tag such as:

```dotenv
APP_IMAGE=ghcr.io/mbonatte/tennis:sha-1a2b3c4
```

Release tags publish `1.2.3`, `1.2`, and `1`; commit tags publish `sha-<short-sha>`. Pinning a commit or full release makes deployments and rollback deterministic.

GHCR packages may initially be private. Either make the package public in GitHub package settings or log in once on the VPS with a classic PAT that has only `read:packages` (and repository access when the source repository is private):

```bash
export GHCR_USER=your-github-user
read -rsp "GHCR token: " GHCR_TOKEN; echo
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
unset GHCR_TOKEN
```

Do not put registry tokens in `.env`. If pulls return `denied`, check package visibility, PAT scope, SSO authorization, image spelling, and `docker logout ghcr.io` followed by a fresh login.

## Models and persistent data

Place legally obtained weights directly in `deploy/models/`; it is mounted read-only at `/app/models`. Required names and checksums are in the repository's `models/README.md`. Weights are never included in the image or deployment commit.

Named volumes persist across container replacement:

- `tennis-analyzer_postgres-data`: database records.
- `tennis-analyzer_redis-data`: Redis append-only queue state.
- `tennis-analyzer_job-data`: uploads, analyzed videos, JSON, plots, and point clips.

`docker compose down` stops the stack without deleting these volumes. **Do not casually run `docker compose down -v`; `-v` deletes the named-volume data.**

## Update and rollback

Update to a tested immutable tag:

```bash
# Edit APP_IMAGE in .env
docker compose pull
docker compose run --rm migrate
docker compose up -d
docker compose ps
```

Before an update, back up PostgreSQL and job data. A minimal logical database backup is:

```bash
mkdir -p backups
docker compose exec -T postgres pg_dump -U tennis -d tennis -Fc > "backups/tennis-$(date +%F-%H%M).dump"
```

Also back up the `job-data` volume with your normal volume/snapshot tooling. Redis can reconstruct some queue state but is not a substitute for PostgreSQL and job-data backups.

Rollback application code by setting `APP_IMAGE` to the previous `sha-*` or release tag, then running `docker compose pull && docker compose up -d`. Database downgrades are intentionally not automatic: review the Alembic migration and restore the pre-upgrade database backup if a release introduced an incompatible schema.

## Troubleshooting

- **Web unhealthy:** inspect `docker compose logs web`, then `postgres` and `redis`; verify `.env` JSON list syntax, `ALLOWED_HOSTS`, PostgreSQL credentials, and `/readyz`.
- **Worker unhealthy:** inspect `docker compose logs worker`; confirm `REDIS_URL`, PostgreSQL settings, memory availability, and model permissions. Keep `WORKER_CONCURRENCY=1` unless RAM/GPU capacity has been measured.
- **Migration failure:** run `docker compose run --rm migrate` without `-d`, verify PostgreSQL health and credentials, and inspect `docker compose logs postgres`. Do not start a newer app image against a migration that failed.
- **Permission errors:** model files must be readable by UID 10001. Named job data is initialized by the same non-root image user.
- **Pull failure:** confirm GHCR login/package visibility and that `APP_IMAGE` names an existing tag.
- **Disk pressure:** inspect `docker system df` and named-volume sizes. Delete jobs through the application; do not delete arbitrary files from a live job directory.

### PostgreSQL password mismatch after first initialization

The official PostgreSQL image applies `POSTGRES_PASSWORD` only when creating an empty database volume. Editing `.env` later does not change the password already stored in PostgreSQL. To preserve existing data, update the role interactively to the current `.env` password:

```bash
docker compose exec postgres psql -U tennis -d tennis
\password tennis
# Enter the current POSTGRES_PASSWORD twice, then:
\q

docker compose restart web worker
docker compose ps
```

Alternatively, restore the original password in `.env`. For a brand-new deployment with no data worth preserving, you may stop the stack and delete only its PostgreSQL volume before starting again. That is destructive; verify the exact volume with `docker volume ls` and never use `docker compose down -v` when job or Redis data must be retained.
