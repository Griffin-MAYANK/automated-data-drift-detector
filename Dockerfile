FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app

WORKDIR /app

# Build and install only the packaged application; runtime data and reports
# are supplied through the /app/data and /app/reports volume mounts.
COPY pyproject.toml README.md /tmp/drift-detector/
COPY src /tmp/drift-detector/src
RUN python -m pip install /tmp/drift-detector \
    && rm -rf /tmp/drift-detector

ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid "${APP_GID}" drift-detector \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
        --home-dir /app --no-create-home --shell /usr/sbin/nologin \
        drift-detector \
    && mkdir -p /app/data /app/reports \
    && chown -R drift-detector:drift-detector /app

USER drift-detector:drift-detector

ENTRYPOINT ["drift-detector"]