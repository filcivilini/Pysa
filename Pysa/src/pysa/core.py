"""
Pysa: PISA-style weighted least squares with BRR-Fay and plausible values.

This implementation follows the methodology described in:
- PISA Data Analysis Manual: SPSS (OECD, 2009)
- PISA 2022 Technical Report (OECD, 2024)

Key assumptions for *actual PISA data*:
- Full student weight column: W_FSTUWT
- Student replicate weights: 80 BRR–Fay replicates, e.g.
    - PISA 2022: W_FSTURWT1 ... W_FSTURWT80
    - PISA 2000–2018: W_FSTR1 ... W_FSTR80
- Fay factor k = 0.5  (do NOT change this for official PISA data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class PysaResult:
    """Container for one regression model result."""
    coef: pd.Series          # combined coefficients across PVs
    se: pd.Series            # combined standard errors across PVs
    r2: float                # mean R^2 across PVs (point estimate only)
    n_obs: int               # number of observations used (min across PVs)
    dep_root: str            # e.g. "MATH", "READ"


def _build_design_matrix(
    df: pd.DataFrame,
    num_vars: Sequence[str],
    cat_vars: Sequence[str],
    fixed_effects: Sequence[str],
) -> pd.DataFrame:
    """
    Build the design matrix X for regression:
    - numeric variables as is
    - categorical variables one-hot encoded (drop_first=True)
    - fixed_effects also one-hot encoded (drop_first=True)
    Intercept is added later in the WLS estimation.

    Returns
    -------
    X : DataFrame
        Design matrix without intercept.
    """
    used_cols = list(num_vars)
    df_work = df.copy()

    all_cat_fe = list(dict.fromkeys(list(cat_vars) + list(fixed_effects)))

    if all_cat_fe:
        df_work = pd.get_dummies(df_work, columns=all_cat_fe, drop_first=True)

    for col in df_work.columns:
        if (col in num_vars) or any(col.startswith(f"{fe}_") for fe in fixed_effects) or any(
            col.startswith(f"{cv}_") for cv in cat_vars
        ):
            if col not in used_cols:
                used_cols.append(col)

    X = df_work[used_cols].copy()
    X = X.astype(float)
    return X


def _wls_brr_fay_single_pv(
    y: pd.Series,
    X: pd.DataFrame,
    base_weights: pd.Series,
    replicate_weights: pd.DataFrame,
    fay_rho: float,
) -> tuple[pd.Series, pd.Series, float, int]:
    """
    Run WLS for a single plausible value using full and replicate weights.

    Returns
    -------
    beta_full : pd.Series
        Coefficients from full-sample WLS.
    var_brr : pd.Series
        BRR-Fay sampling variance for each coefficient.
    r2_full : float
        R-squared from full-sample WLS.
    n_obs : int
        Number of observations used.
    """
    merged = pd.concat([y, X, base_weights, replicate_weights], axis=1).dropna()
    y_use = merged[y.name]
    X_use = merged[X.columns]
    w_full = merged[base_weights.name]

    X_with_const = sm.add_constant(X_use, has_constant="add")

    model_full = sm.WLS(y_use, X_with_const, weights=w_full).fit()
    beta_full = model_full.params
    r2_full = float(model_full.rsquared)
    n_obs = int(model_full.nobs)

    betas_rep = []
    for col in replicate_weights.columns:
        w_rep = merged[col]
        model_rep = sm.WLS(y_use, X_with_const, weights=w_rep).fit()
        betas_rep.append(model_rep.params)

    betas_rep = pd.DataFrame(betas_rep)

    G = replicate_weights.shape[1]
    constant = 1.0 / (G * (1.0 - fay_rho) ** 2)

    diff = betas_rep.sub(beta_full, axis=1)
    var_brr = constant * (diff ** 2).sum(axis=0)

    return beta_full, var_brr, r2_full, n_obs


def pysa_wls(
    data: pd.DataFrame,
    dep_root: str,
    n_pv: int,
    num_vars: Sequence[str],
    cat_vars: Sequence[str],
    fixed_effects: Sequence[str] = (),
    weight_col: str = "W_FSTUWT",
    replicate_weight_prefix: str = "W_FSTURWT",  # use "W_FSTR" for older cycles
    n_replicates: int = 80,
    fay_rho: float = 0.5,
    country_var: str = "CNTRYID",
    country_values: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> PysaResult:
    """
    Run PISA-style WLS regression with BRR-Fay and plausible values.

    Parameters
    ----------
    data : DataFrame
        PISA-like student dataset.
    dep_root : str
        Root of the plausible value variable names, e.g. "MATH" -> PV1MATH..PVnMATH.
    n_pv : int
        Number of plausible values to use (e.g. 10 for PISA 2022).
    num_vars : sequence of str
        Names of numeric regressors (treated as continuous).
    cat_vars : sequence of str
        Names of categorical regressors (one-hot encoded).
    fixed_effects : sequence of str, optional
        Variables to be treated as fixed effects (also one-hot encoded).
    weight_col : str, default "W_FSTUWT"
        Full student weight variable.
    replicate_weight_prefix : str, default "W_FSTURWT"
        Prefix of replicate weight columns (e.g. "W_FSTR" or "W_FSTURWT").
    n_replicates : int, default 80
        Number of replicate weights (80 in PISA).
    fay_rho : float, default 0.5
        Fay factor k. For official PISA data this must be 0.5.
    country_var : str, default "CNTRYID"
        Country identifier variable.
    country_values : int, str, or iterable, optional
        Value(s) of country_var to keep (e.g. 380 for Italy). If None, no filter.

    Returns
    -------
    PysaResult
        Contains coefficients, standard errors, R² and n_obs.

    Notes
    -----
    For *official* PISA use, the following should be used:
    - weight_col="W_FSTUWT"
    - n_replicates=80
    - fay_rho=0.5
    - replicate_weight_prefix:
        * "W_FSTR" for older cycles (e.g. PISA 2009)
        * "W_FSTURWT" for PISA 2022 (and similar if documented)
    """
    df = data.copy()

    # Optional: clearer error messages for regressors
    missing_regressors = (
        [c for c in num_vars if c not in df.columns]
        + [c for c in cat_vars if c not in df.columns]
        + [c for c in fixed_effects if c not in df.columns]
    )
    if missing_regressors:
        raise ValueError(f"Missing regressor columns: {missing_regressors}")

    if country_values is not None:
        if isinstance(country_values, (int, str)):
            df = df[df[country_var] == country_values]
        else:
            df = df[df[country_var].isin(list(country_values))]

    pv_cols = [f"PV{i}{dep_root}" for i in range(1, n_pv + 1)]
    missing_pv = [c for c in pv_cols if c not in df.columns]
    if missing_pv:
        raise ValueError(f"Missing plausible value columns: {missing_pv}")

    if weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in data.")

    rep_cols = [f"{replicate_weight_prefix}{i}" for i in range(1, n_replicates + 1)]
    missing_rep = [c for c in rep_cols if c not in df.columns]
    if missing_rep:
        raise ValueError(f"Missing replicate weight columns: {missing_rep}")

    X = _build_design_matrix(df, num_vars=num_vars, cat_vars=cat_vars, fixed_effects=fixed_effects)

    base_w = df[weight_col]
    rep_w = df[rep_cols]

    betas_full = []
    vars_brr = []
    r2_list = []
    n_list = []

    for pv in pv_cols:
        y = df[pv]
        beta_full, var_brr, r2_full, n_obs = _wls_brr_fay_single_pv(
            y=y,
            X=X,
            base_weights=base_w,
            replicate_weights=rep_w,
            fay_rho=fay_rho,
        )
        betas_full.append(beta_full)
        vars_brr.append(var_brr)
        r2_list.append(r2_full)
        n_list.append(n_obs)

    betas_df = pd.DataFrame(betas_full)
    vars_df = pd.DataFrame(vars_brr)

    M = len(betas_df)

    beta_bar = betas_df.mean(axis=0)
    U_bar = vars_df.mean(axis=0)

    if M > 1:
        B = betas_df.sub(beta_bar, axis=1).pow(2).sum(axis=0) / (M - 1)
    else:
        B = 0.0

    total_var = U_bar + (1.0 + 1.0 / M) * B
    se = total_var.pow(0.5)

    r2_mean = float(np.mean(r2_list))
    n_obs_out = int(min(n_list)) if n_list else 0

    return PysaResult(
        coef=beta_bar,
        se=se,
        r2=r2_mean,
        n_obs=n_obs_out,
        dep_root=dep_root,
    )


