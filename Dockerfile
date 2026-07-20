# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime
ARG TORCH_INDEX_URL
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 tini && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 10001 tennis && useradd --uid 10001 --gid tennis --create-home tennis
WORKDIR /app
COPY pyproject.toml README.md ./
RUN python -m pip install --upgrade pip==25.1.1 \
    && if [ -n "${TORCH_INDEX_URL:-}" ]; then python -m pip install --index-url "$TORCH_INDEX_URL" torch==2.8.0 torchvision==0.23.0; fi \
    && python -c "import subprocess, tomllib; project = tomllib.load(open('pyproject.toml', 'rb'))['project']; requirements = [item for item in project['dependencies'] + project['optional-dependencies']['ml'] if not item.startswith(('torch==', 'torchvision=='))]; subprocess.check_call(['python', '-m', 'pip', 'install', *requirements])"
COPY app ./app
COPY tennis_analyzer ./tennis_analyzer
COPY alembic ./alembic
COPY alembic.ini ./
COPY analysis.py ball.py bounce_detector.py court.py court_detection_net.py court_reference.py homography.py main.py player.py postprocess.py tracking_postprocess.py tracknet.py ./
COPY BallTrack ./BallTrack
RUN python -m pip install --no-deps ".[ml]"
RUN mkdir -p /app/data/jobs /app/models && chown -R tennis:tennis /app
USER tennis
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)" || exit 1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
