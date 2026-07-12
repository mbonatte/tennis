from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from tennis_analyzer.config import ModelPaths

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    models = ModelPaths.from_root(settings.model_root)
    available = {name: path.is_file() for name, path in vars(models).items() if name != "root"}
    missing = [name for name, present in available.items() if not present]
    if missing:
        logger.warning("Optional model files missing at startup: %s", ", ".join(missing))
    else:
        logger.info("All configured model files are present")
    yield


app = FastAPI(title="Tennis Analyzer", version="0.1.0", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(router)
