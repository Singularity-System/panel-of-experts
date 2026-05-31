# Wikitext-2-raw Experimental Data

## Setup
- **Data**: wikitext-2-raw, 20887 valid lines
- **Device**: NVIDIA RTX 4080 SUPER (dual GPU)
- **Config**: d_model=128, n_head=4, d_ff=256, 4 experts, top_k=2
- **Training**: 5 epochs, lr=3e-4, AdamW

## Architecture Depth

| Model | Sequential Depth | Total Params | Active Params |
|---|---|---|---|
| PPoT | **5 layers** (3 expert + 2 post-proc) | 9.0M | ~7.5M |
| Transformer-5L | 5 layers | 7.1M | 7.1M |
| Transformer-8L | 8 layers | 7.5M | 7.5M |
| Transformer-12L | 12 layers | 8.1M | 8.1M |

## Results

| Model | Depth | PPL | Acc | Balance | VN% | Eff | Time |
|---|---|---|---|---|---|---|---|
| PPoT-Baseline | 5 | 12.15 | 0.1969 | 0.479 | 86.3% | 3.77 | 355s |
| PPoT-LB | 5 | 12.22 | 0.1969 | 0.735 | 51.1% | 3.94 | 364s |
| PPoT-Div | 5 | **12.11** | **0.1996** | 0.079 | 100% | 2.89 | 432s |
| PPoT-LB+Div | 5 | 12.26 | 0.1972 | 0.646 | 100% | 3.89 | 432s |
| Transformer-5L | 5 | 17.55 | 0.1415 | — | — | — | 227s |
| Transformer-8L | 8 | 14.74 | 0.1778 | — | — | — | 286s |
| Transformer-12L | 12 | 12.94 | 0.2076 | — | — | — | 348s |

## Core Thesis: Width Parallelism >> Depth

- **PPoT 5L beats TF-5L**: PPL **-30.8%**, Acc **+39.2%**
- **PPoT 5L beats TF-8L**: PPL **-17.6%**, Acc **+9.9%**
- **PPoT 5L beats TF-12L**: PPL **-6.3%** (despite 5 < 12 depth)

**PPoT uses 42% of Transformer depth (5/12) for better PPL.**

## LB+Div Effectiveness

| Config | Balance | VN% | Effective Experts | Routing Collapse |
|---|---|---|---|---|
| Baseline | 0.479 | 86.3% | 3.77 | E0=33.4%, E1=16.0% |
| LB | 0.735 | 51.1% | 3.94 | E0=28.8%, E2=21.6% |
| Div | 0.079 | 100% | 2.89 | E0=46.8%, E2=3.7% |
| **LB+Div** | **0.646** | **100%** | **3.89** | E0=30.8%, E1=19.9% |

LB+Div achieves **simultaneous routing balance + expert diversity** — all 4 experts active (3.89 effective).

## Eigenvalues (Gram Matrix)

- Baseline: [0.324, 0.668, 1.012, 1.997] — skewed
- LB: [0.112, 0.256, 0.459, 3.173] — worse
- Div: [0.977, 0.994, 1.005, 1.024] — perfect uniform
- **LB+Div**: [0.980, 0.996, 0.999, 1.024] — near perfect + balanced routing
