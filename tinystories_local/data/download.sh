#!/bin/bash
BASE="https://huggingface.co/datasets/Salesforce/tiny_stories/resolve/main/data"
for f in train-00000-of-00004 train-00001-of-00004 train-00002-of-00004 train-00003-of-00004 validation-00000-of-00001; do
  echo "Downloading $f..."
  curl -L -k --connect-timeout 10 --max-time 120 -o "${f}.parquet" "${BASE}/${f}.parquet" && echo "OK: $f" || echo "FAIL: $f"
done
