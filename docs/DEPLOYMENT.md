# Deployment architecture

Production deployment is pull-only. GitHub Actions builds `Dockerfile`, scans the result, and publishes it to `ghcr.io/mbonatte/tennis`. The self-contained [`deploy/`](../deploy/) directory is the only application bundle needed on the VPS; its Compose file contains no `build:` directives.

The production sequence is:

```bash
cd deploy
cp .env.example .env
docker compose pull
docker compose run --rm migrate
docker compose up -d
```

Web, worker, and the explicit migration job use the same `APP_IMAGE`. PostgreSQL, Redis AOF, and job data use named volumes; models use a read-only `deploy/models` bind mount. PostgreSQL and Redis are isolated on an internal network. Only the web service joins the existing Nginx Proxy Manager `proxy` network.

Use an immutable `sha-*` or release tag for production and rollback. Database migration is never performed by web or worker startup. See [deploy/README.md](../deploy/README.md) for exact first-deploy, registry authentication, model setup, backup, update, rollback, shutdown, and troubleshooting procedures.
