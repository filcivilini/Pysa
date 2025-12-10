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

What this version adds:
- Standardized coefficients, SEs, and t-values.
  These are computed using weighted SDs (full weight) on the same analytic
  sample used for each PV regression.
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

    # Unstandardized (combined across PVs)
    coef: pd.Series
    se: pd.Series

    # Standardized (combined across PVs)
    std_coef: pd.Series
    std_se: pd.Series

    # Model-level info
    r2: float
    n_obs: int
    dep_root: str

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy DataFrame with unstandardized and standardized results."""
        df = pd.DataFrame({
            "Variable": self.coef.index,
            "coef": self.coef.values,
            "se": self.se.reindex(self.coef.index).values,
            "std_coef": self.std_coef.reindex(self.coef.index).values,
            "std_se": self.std_se.reindex(self.coef.index).values,
        })
        df["t"] = df["coef"] / df["se"]
        df["std_t"] = df["std_coef"] / df["std_se"]
        return df



def _wmean(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(w, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(w)
    x = x[mask]
    w = w[mask]
    if w.size == 0 or np.sum(w) == 0:
        return float("nan")
    return float(np.sum(w * x) / np.sum(w))


def _wsd(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(w, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(w)
    x = x[mask]
    w = w[mask]
    if w.size == 0 or np.sum(w) == 0:
        return float("nan")
    m = np.sum(w * x) / np.sum(w)
    var = np.sum(w * (x - m) ** 2) / np.sum(w)
    return float(np.sqrt(var))



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
    compute_standardized: bool = True,
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
    compute_standardized : bool, default True
        Whether to compute standardized coefficients/SEs.

    Returns
    -------
    PysaResult
        Contains unstandardized and standardized coefficients, standard errors,
        mean R² and n_obs.

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

    # Check regressors exist
    missing_regressors = (
        [c for c in num_vars if c not in df.columns]
        + [c for c in cat_vars if c not in df.columns]
        + [c for c in fixed_effects if c not in df.columns]
    )
    if missing_regressors:
        raise ValueError(f"Missing regressor columns: {missing_regressors}")

    # Country filter
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

    # Build X once
    X = _build_design_matrix(df, num_vars=num_vars, cat_vars=cat_vars, fixed_effects=fixed_effects)

    base_w = df[weight_col]
    rep_w = df[rep_cols]

    # Storage per PV
    betas_full = []
    vars_brr = []
    r2_list = []
    n_list = []

    betas_std = []
    vars_std = []

    for pv in pv_cols:
        y = df[pv]

        # Unstandardized WLS + BRR-Fay
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

        # Standardized (optional)
        if compute_standardized:
            # Use the same analytic sample definition as the single-PV function
            merged = pd.concat([y, X, base_w, rep_w], axis=1).dropna()
            y_use = merged[pv]
            X_use = merged[X.columns]
            w_full = merged[weight_col]

            sd_y = _wsd(y_use, w_full)

            scale = {}
            for name in beta_full.index:
                if name == "const":
                    scale[name] = np.nan
                elif name in X_use.columns:
                    sd_x = _wsd(X_use[name], w_full)
                    scale[name] = (sd_x / sd_y) if (sd_y and np.isfinite(sd_y) and sd_y != 0) else np.nan
                else:
                    scale[name] = np.nan

            scale_s = pd.Series(scale)

            beta_std = beta_full * scale_s
            var_std_brr = var_brr * (scale_s ** 2)

            betas_std.append(beta_std)
            vars_std.append(var_std_brr)

    # Combine across PVs (Rubin-style)
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

    # Standardized combine
    if compute_standardized and betas_std:
        betas_std_df = pd.DataFrame(betas_std)
        vars_std_df = pd.DataFrame(vars_std)

        beta_bar_std = betas_std_df.mean(axis=0)
        U_bar_std = vars_std_df.mean(axis=0)

        if M > 1:
            B_std = betas_std_df.sub(beta_bar_std, axis=1).pow(2).sum(axis=0) / (M - 1)
        else:
            B_std = 0.0

        total_var_std = U_bar_std + (1.0 + 1.0 / M) * B_std
        se_std = total_var_std.pow(0.5)
    else:
        # Provide empty standardized outputs aligned with unstandardized index
        beta_bar_std = pd.Series(index=beta_bar.index, dtype="float64")
        se_std = pd.Series(index=beta_bar.index, dtype="float64")

    r2_mean = float(np.mean(r2_list)) if r2_list else float("nan")
    n_obs_out = int(min(n_list)) if n_list else 0

    # Ensure aligned indices
    beta_bar_std = beta_bar_std.reindex(beta_bar.index)
    se_std = se_std.reindex(beta_bar.index)

    return PysaResult(
        coef=beta_bar,
        se=se,
        std_coef=beta_bar_std,
        std_se=se_std,
        r2=r2_mean,
        n_obs=n_obs_out,
        dep_root=dep_root,
    )
