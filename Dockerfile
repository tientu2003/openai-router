# --------------------------------------------------------------
# Dockerfile for the OpenRouter FastAPI proxy
# --------------------------------------------------------------

# Use the official slim Python 3.13 image (matches the local interpreter)
FROM python:3.13-slim

# ---------- Build‑time metadata ----------
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0
LABEL \
    org.opencontainers.image.title="OpenRouter FastAPI Proxy" \
    org.opencontainers.image.description="A thin FastAPI proxy for OpenRouter/Groq" \
    org.opencontainers.image.version="${VERSION}" \
    org.opencontainers.image.created="${BUILD_DATE}" \
    org.opencontainers.image.revision="${VCS_REF}" \
    org.opencontainers.image.authors="AI Assistant"

# ---------- System dependencies ----------
# Install a minimal set of OS packages needed for building wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc && \
    rm -rf /var/lib/apt/lists/*

# ---------- Create a non‑root user ----------
# uid/gid 1000 is a common default for a regular user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --create-home --shell /bin/bash appuser

# ---------- Application files ----------
WORKDIR /app

# Copy only the files required for installing dependencies first.
# This improves Docker layer caching when source code changes.
COPY requirements.txt .

# Install Python dependencies in a virtual environment inside the container.
# Using `--no-cache-dir` keeps the image size small.
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code.
COPY . .

# Ensure the non‑root user owns the application files.
RUN chown -R appuser:appgroup /app

# ---------- Runtime configuration ----------
# Use the virtual environment's python and uvicorn.
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose the default FastAPI/uvicorn port.
EXPOSE 8000

# Switch to the non‑root user.
USER appuser

# Entry point: run the FastAPI app with uvicorn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]