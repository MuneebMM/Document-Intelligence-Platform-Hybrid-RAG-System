#!/bin/bash
set -e

echo "Starting Document Intelligence Platform Backend..."
echo "Environment: $ENVIRONMENT"

# Ensure the BM25 index persistence directory exists.
# Created here rather than in application code so it is always present
# even on a cold start before any document has been ingested.
mkdir -p bm25_indexes

exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 2 \
  --timeout-keep-alive 600 \
  --log-level info
