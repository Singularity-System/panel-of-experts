#!/bin/bash
# Run PPoT vs Transformer benchmark on Wikitext-2
# Usage: bash run_wikitext2_benchmark.sh [--samples 50000] [--epochs 5] [--d_model 128]

set -e

echo "=== Installing dependencies ==="
pip install transformers datasets accelerate -q 2>/dev/null

echo "=== Running benchmark ==="
python3 run_wikitext2_benchmark.py "$@"

echo ""
echo "=== Done ==="
echo "Results include: PPL, Accuracy, Params, Balance, Von Neumann Entropy, Routing Stats, Eigenvalues"
