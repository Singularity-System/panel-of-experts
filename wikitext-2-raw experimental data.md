# Wikitext-2-raw Experimental Data

## Setup
- **Data**: wikitext-2-raw, 20887 valid lines (from 36718 total)
- **Device**: NVIDIA RTX 4080 SUPER (dual GPU)
- **Config**: d_model=128, n_head=4, d_ff=256, 4 experts, top_k=2
- **Training**: 5 epochs, lr=3e-4, AdamW
- **LB α=0.1, Div α=0.5**

## Results

| Model | PPL | Acc | Params | Balance | VN% | Dead | Eff | Time |
|---|---|---|---|---|---|---|---|---|
| PPoT-Baseline | **12.48** | **0.1983** | 9.0M | 0.506 | 83.4% | 0 | 3.71 | 361s |
| PPoT-LB | 12.61 | 0.1946 | 9.0M | 0.728 | 62.6% | 0 | 3.92 | 364s |
| PPoT-Div | 12.48 | 0.1967 | 9.0M | 0.073 | 100.0% | 0 | 2.88 | 428s |
| PPoT-LB+Div | 12.60 | 0.1952 | 9.0M | 0.617 | 99.9% | 0 | 3.89 | 429s |
| Transformer-5L | 17.50 | 0.1463 | 7.1M | — | — | — | — | 229s |
| Transformer-8L | 14.77 | 0.1809 | 7.5M | — | — | — | — | 286s |
| Transformer-12L | 13.30 | **0.2065** | 8.1M | — | — | — | — | 347s |

## Key Findings

1. **PPoT beats shallow Transformers**: vs TF-5L PPL -28.7%, Acc +35.5%
2. **PPoT beats moderate Transformers**: vs TF-8L PPL -15.5%, Acc +9.5%
3. **TF-12L narrows the gap**: PPL +6.6% worse but Acc +4.1% better than PPoT
4. **LB+Div combined**: best routing balance (0.617) + perfect expert diversity (99.9%)
5. **Div alone causes routing collapse**: balance=0.073, E2 gets only 5989 routes

## Routing Details (PPoT)

| Config | E0 | E1 | E2 | E3 | Balance | Eff |
|---|---|---|---|---|---|---|
| Baseline | 28747 | 56825 | 53724 | 34050 | 0.506 | 3.71 |
| LB | 50321 | 37762 | 36610 | 48653 | 0.728 | 3.92 |
| Div | 81622 | 48654 | 5989 | 37081 | 0.073 | 2.88 |
| LB+Div | 53029 | 32706 | 44584 | 43027 | 0.617 | 3.89 |

## Eigenvalues (Gram Matrix)

- Baseline: [0.377, 0.561, 0.859, 2.204]
- LB: [0.133, 0.412, 0.586, 2.868]
- Div: [0.977, 0.979, 1.004, 1.040] ← perfect uniform
- LB+Div: [0.954, 0.975, 0.996, 1.074] ← near perfect
