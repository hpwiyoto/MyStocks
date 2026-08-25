# Single image reused for both the scheduler (pipeline/features/engine) and
# the Streamlit app -- avoids duplicating dependency installation across two
# Dockerfiles. Which process runs is decided by docker-compose's `command`.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
