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

# --preload loads the ~10 MB nominations dataset and the precomputed edge table
# once in the master process and forks the workers, instead of paying it per worker.
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "3", "--preload", "wsgi:application"]
