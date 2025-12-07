# pysa

**pysa** provides PISA-style weighted least squares (WLS) with **BRR-Fay replicate weights** and **plausible values**.

It is designed for analyses aligned with standard OECD/PISA practices:
- full-sample student weights
- 80 BRR–Fay replicate weights
- plausible value combination using Rubin-style rules (sampling + imputation variance)

> This package is intended for research and reproducible workflows using PISA-like datasets.

---

## Features

- **Weighted least squares (WLS)** using full-sample weights
- **BRR-Fay variance estimation** using replicate weights
- **Plausible value (PV) combination**
  - Averages coefficients across PVs
  - Combines sampling + between-PV variance into final standard errors
- Flexible model specification:
  - numeric regressors (`num_vars`)
  - categorical regressors (`cat_vars`, one-hot encoded with `drop_first=True`)
  - fixed effects (`fixed_effects`, also one-hot encoded)
- Optional **country filtering**
- Clean results object: `PysaResult`

---

## Installation

### From PyPI (once published)
```bash
pip install pysa
