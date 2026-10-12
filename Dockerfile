FROM python:3.12-slim

WORKDIR /app

# uv for fast, lock-consistent installs
RUN pip install --no-cache-dir uv

# Dependencies in their own layer — only rebuilt when the lock changes.
# requirements.txt is generated from uv.lock, never edited by hand:
#   uv export --format requirements-txt --no-dev --no-hashes -o requirements.txt
COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# App code + data files
COPY . .

EXPOSE 8050

# No --preload: polars deadlocks in a forked child once its thread pool has run in
# the parent (and building df_edges at import runs it), so every network/nominations
# callback would hang until the worker timeout. Each worker loads the app itself
# instead (~450 MB RSS each). gthread keeps a few slow requests (bubbles ~2.5 s)
# from blocking the app; callbacks share no mutable state.
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "--worker-class", "gthread", "--threads", "4", "wsgi:application"]
