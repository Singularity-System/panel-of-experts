# Wikitext-2-raw Experimental Data

## Setup
- **Data**: wikitext-2-raw, 20887 valid lines
- **Device**: NVIDIA RTX 4080 SUPER (dual GPU)
- **Config**: d_model=128, n_head=4, d_ff=256, 4 experts, top_k=2
- **Training**: 5 epochs, lr=3e-4, AdamW

## Architecture Depth

| Model | Sequential Depth | Total Params | Active Params (per forward) |
|---|---|---|---|
| PPoT | **5 layers** (3 expert + 2 post-proc) | 9.0M | ~7.5M (top_k=2) |
| Transformer-5L | 5 layers | 7.1M | 7.1M |
| Transformer-8L | 8 layers | 7.5M | 7.5M |
| Transformer-12L | 12 layers | 8.1M | 8.1M |

## Results

| Model | Depth | PPL | Acc | Time |
|---|---|---|---|---|
| PPoT-Baseline | **5** | **12.48** | 0.1983 | 361s |
| PPoT-LB | 5 | 12.61 | 0.1946 | 364s |
| PPoT-Div | 5 | 12.48 | 0.1967 | 428s |
| PPoT-LB+Div | 5 | 12.60 | 0.1952 | 429s |
| Transformer-5L | 5 | 17.50 | 0.1463 | 229s |
| Transformer-8L | 8 | 14.77 | 0.1809 | 286s |
| Transformer-12L | 12 | 13.30 | 0.2065 | 347s |

## Key Findings

1. **PPoT (5L) beats Transformer-5L by 28.7% PPL** — same depth, MoE wins big
2. **PPoT (5L) beats Transformer-8L by 15.5% PPL** — 5L beats 8L
3. **PPoT (5L) beats Transformer-12L by 6.2% PPL** — 5L beats 12L
4. **PPoT uses 43% depth for better PPL than 12L Transformer**
5. **TF-12L only wins on Acc (+4.1%)** — depth helps accuracy, but PPoT wins on probability calibration (PPL)

## Routing Details

| Config | E0 | E1 | E2 | E3 | Balance | VN% | Eff |
|---|---|---|---|---|---|---|---|
| Baseline | 28747 | 56825 | 53724 | 34050 | 0.506 | 83.4% | 3.71 |
| LB | 50321 | 37762 | 36610 | 48653 | 0.728 | 62.6% | 3.92 |
| Div | 81622 | 48654 | 5989 | 37081 | 0.073 | 100.0% | 2.88 |
| LB+Div | 53029 | 32706 | 44584 | 43027 | 0.617 | 99.9% | 3.89 |

## Eigenvalues

- Baseline: [0.377, 0.561, 0.859, 2.204]
- LB: [0.133, 0.412, 0.586, 2.868]
- Div: [0.977, 0.979, 1.004, 1.040]
- LB+Div: [0.954, 0.975, 0.996, 1.074]
