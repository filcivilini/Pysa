from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .replication import ReplicationDesign, sampling_variance
from .results import PysaEstimate, PysaResult
from .estimators import WlsSpec, wsd


def _rubin_combine(betas: pd.DataFrame, vars_: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    M = int(betas.shape[0])
    beta_bar = betas.mean(axis=0)
    U_bar = vars_.mean(axis=0)

    if M > 1:
        B = betas.sub(beta_bar, axis=1).pow(2).sum(axis=0) / (M - 1)
    else:
        B = 0.0

    total_var = U_bar + (1.0 + 1.0 / M) * B
    se = total_var.pow(0.5)
    return beta_bar, se


def _subset_for_design(df: pd.DataFrame, design: ReplicationDesign, needed: Sequence[str]) -> pd.DataFrame:
    cols = list(dict.fromkeys(list(needed)))
    if design.replicate_cols is not None:
        cols += [c for c in design.replicate_cols if c not in cols]
    else:
        cols += [c for c in (design.zone_col, design.rep_col) if c is not None and c not in cols]
    cols += [design.weight_col] if design.weight_col not in cols else []
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[cols].copy()


def run_with_pvs(
    data: pd.DataFrame,
    pv_cols: Sequence[str],
    replication: ReplicationDesign,
    estimator: Callable[[pd.DataFrame, pd.Series], pd.Series],
    required_cols: Optional[Sequence[str]] = None,
) -> PysaEstimate:
    if not pv_cols:
        raise ValueError("pv_cols must be a non-empty sequence.")

    estimates: list[pd.Series] = []
    variances: list[pd.Series] = []

    for pv in pv_cols:
        needed = [pv]
        if required_cols is not None:
            needed += list(required_cols)

        df_pv = _subset_for_design(data, replication, needed=needed).dropna()
        replication.validate(df_pv)

        w_full = df_pv[replication.weight_col]
        theta_full = estimator(df_pv, w_full)

        rep_thetas = []
        for w_rep in replication.iter_replicate_weights(df_pv, w_full):
            rep_thetas.append(estimator(df_pv, w_rep))
        thetas_rep = pd.DataFrame(rep_thetas)

        scale = replication.replicate_scale(n_reps=int(thetas_rep.shape[0]))
        var_samp = sampling_variance(theta_full, thetas_rep, scale=scale)

        estimates.append(theta_full)
        variances.append(var_samp)

    est_df = pd.DataFrame(estimates)
    var_df = pd.DataFrame(variances)

    est, se = _rubin_combine(est_df, var_df)
    return PysaEstimate(estimate=est, se=se)


def pysa_mean(
    data: pd.DataFrame,
    pv_cols: Sequence[str],
    replication: ReplicationDesign,
) -> PysaEstimate:
    from .estimators import mean_estimator
    return run_with_pvs(
        data=data,
        pv_cols=pv_cols,
        replication=replication,
        estimator=mean_estimator(pv_cols),
        required_cols=None,
    )


def pysa_proportion(
    data: pd.DataFrame,
    pv_cols: Sequence[str],
    replication: ReplicationDesign,
    cutoff: float,
    ge: bool = True,
) -> PysaEstimate:
    from .estimators import proportion_estimator
    return run_with_pvs(
        data=data,
        pv_cols=pv_cols,
        replication=replication,
        estimator=proportion_estimator(pv_cols, cutoff=cutoff, ge=ge),
        required_cols=None,
    )

def pysa_correlation_with_pv(
    data: pd.DataFrame,
    pv_cols: Sequence[str],
    replication: ReplicationDesign,
    other_col: str,
) -> PysaEstimate:
    from .estimators import correlation
    est = correlation(pv_cols=pv_cols, other_col=other_col)
    return run_with_pvs(
        data=data,
        pv_cols=pv_cols,
        replication=replication,
        estimator=est,
        required_cols=[other_col],
    )


def pysa_percentiles(
    data: pd.DataFrame,
    pv_cols: Sequence[str],
    replication: ReplicationDesign,
    probs: Sequence[float],
) -> PysaEstimate:
    from .estimators import percentiles
    est = percentiles(pv_cols=pv_cols, probs=probs)
    return run_with_pvs(
        data=data,
        pv_cols=pv_cols,
        replication=replication,
        estimator=est,
        required_cols=None,
    )


def pysa_cumulative_benchmarks(
    data: pd.DataFrame,
    pv_cols: Sequence[str],
    replication: ReplicationDesign,
    cutpoints: Sequence[float],
) -> PysaEstimate:
    """
    Cumulative benchmarks: P(PV >= cutpoint) for each cutpoint.

    NOTE: Do NOT default cutpoints here; benchmarks are study-specific.
    """
    from .estimators import cumulative_benchmarks
    est = cumulative_benchmarks(pv_cols=pv_cols, cutpoints=cutpoints)
    return run_with_pvs(
        data=data,
        pv_cols=pv_cols,
        replication=replication,
        estimator=est,
        required_cols=None,
    )


def pysa_band_benchmarks(
    data: pd.DataFrame,
    pv_cols: Sequence[str],
    replication: ReplicationDesign,
    cutpoints: Sequence[float],
) -> PysaEstimate:
    """
    Band benchmarks: mutually exclusive bands defined by cutpoints.

    NOTE: Do NOT default cutpoints here; benchmarks are study-specific.
    """
    from .estimators import band_benchmarks
    est = band_benchmarks(pv_cols=pv_cols, cutpoints=cutpoints)
    return run_with_pvs(
        data=data,
        pv_cols=pv_cols,
        replication=replication,
        estimator=est,
        required_cols=None,
    )

def _build_design_matrix(
    df: pd.DataFrame,
    num_vars: Sequence[str],
    cat_vars: Sequence[str],
    fixed_effects: Sequence[str],
) -> pd.DataFrame:
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
    return X.astype(float)


def pysa_wls(
    data: pd.DataFrame,
    dep_root: str,
    n_pv: int,
    num_vars: Sequence[str],
    cat_vars: Sequence[str],
    fixed_effects: Sequence[str] = (),

    weight_col: str = "W_FSTUWT",
    replicate_weight_prefix: str = "W_FSTURWT",
    n_replicates: int = 80,
    fay_rho: float = 0.5,

    pv_cols: Optional[Sequence[str]] = None,
    replication: Optional[ReplicationDesign] = None,

    country_var: str = "CNTRYID",
    country_values: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,

    compute_standardized: bool = True,
) -> PysaResult:
    df = data.copy()

    if country_values is not None:
        if isinstance(country_values, (int, str)):
            df = df[df[country_var] == country_values]
        else:
            df = df[df[country_var].isin(list(country_values))]

    if pv_cols is None:
        pv_cols = [f"PV{i}{dep_root}" for i in range(1, n_pv + 1)]

    missing_pv = [c for c in pv_cols if c not in df.columns]
    if missing_pv:
        raise ValueError(f"Missing plausible value columns: {missing_pv}")

    if replication is None:
        rep_cols = [f"{replicate_weight_prefix}{i}" for i in range(1, n_replicates + 1)]
        replication = ReplicationDesign(
            method="brr_fay",
            weight_col=weight_col,
            replicate_cols=rep_cols,
            fay_rho=fay_rho,
        )

    if replication.weight_col not in df.columns:
        raise ValueError(f"Weight column '{replication.weight_col}' not found in data.")

    missing_reg = (
        [c for c in num_vars if c not in df.columns]
        + [c for c in cat_vars if c not in df.columns]
        + [c for c in fixed_effects if c not in df.columns]
    )
    if missing_reg:
        raise ValueError(f"Missing regressor columns: {missing_reg}")

    X = _build_design_matrix(df, num_vars=num_vars, cat_vars=cat_vars, fixed_effects=fixed_effects)
    X_cols = list(X.columns)

    betas_full: list[pd.Series] = []
    vars_samp: list[pd.Series] = []
    r2_list: list[float] = []
    n_list: list[int] = []

    betas_std: list[pd.Series] = []
    vars_std: list[pd.Series] = []

    for pv in pv_cols:
        base = pd.concat([df[[pv]], X], axis=1)
        base[replication.weight_col] = df[replication.weight_col]

        if replication.replicate_cols is not None:
            base = pd.concat([base, df[list(replication.replicate_cols)]], axis=1)
        else:
            if replication.zone_col is None or replication.rep_col is None:
                raise ValueError("JK2 replication requires zone_col and rep_col.")
            base = pd.concat([base, df[[replication.zone_col, replication.rep_col]]], axis=1)

        base = base.dropna()
        replication.validate(base)

        w_full = base[replication.weight_col]
        spec = WlsSpec(y_col=pv, x_cols=X_cols)
        beta_full = spec.params(base, w_full)

        rep_betas = []
        for w_rep in replication.iter_replicate_weights(base, w_full):
            rep_betas.append(spec.params(base, w_rep))
        betas_rep = pd.DataFrame(rep_betas)

        scale = replication.replicate_scale(n_reps=int(betas_rep.shape[0]))
        var_samp = sampling_variance(beta_full, betas_rep, scale=scale)

        r2, nobs = spec.fit_stats(base, w_full)

        betas_full.append(beta_full)
        vars_samp.append(var_samp)
        r2_list.append(float(r2))
        n_list.append(int(nobs))

        if compute_standardized:
            y_use = base[pv]
            sd_y = wsd(y_use, w_full)

            scale_map = {}
            for name in beta_full.index:
                if name == "const":
                    scale_map[name] = np.nan
                elif name in X_cols:
                    sd_x = wsd(base[name], w_full)
                    if np.isfinite(sd_y) and sd_y != 0:
                        scale_map[name] = sd_x / sd_y
                    else:
                        scale_map[name] = np.nan
                else:
                    scale_map[name] = np.nan

            scale_s = pd.Series(scale_map)
            beta_std = beta_full * scale_s
            var_std_samp = var_samp * (scale_s ** 2)

            betas_std.append(beta_std)
            vars_std.append(var_std_samp)

    betas_df = pd.DataFrame(betas_full)
    vars_df = pd.DataFrame(vars_samp)
    coef, se = _rubin_combine(betas_df, vars_df)

    if compute_standardized and betas_std:
        bstd_df = pd.DataFrame(betas_std)
        vstd_df = pd.DataFrame(vars_std)
        std_coef, std_se = _rubin_combine(bstd_df, vstd_df)
        std_coef = std_coef.reindex(coef.index)
        std_se = std_se.reindex(coef.index)
    else:
        std_coef = pd.Series(index=coef.index, dtype="float64")
        std_se = pd.Series(index=coef.index, dtype="float64")

    return PysaResult(
        coef=coef,
        se=se,
        std_coef=std_coef,
        std_se=std_se,
        r2=float(np.mean(r2_list)) if r2_list else float("nan"),
        n_obs=int(min(n_list)) if n_list else 0,
        dep_root=dep_root,
    )

