from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Iterable, Union, Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _to_float_array(x: pd.Series) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)


def _pick_present_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """
    Return the first column name in `candidates` that exists in df.
    This is crucial for run_with_pvs(), which subsets only the current PV column.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns are present: {list(candidates)}")


def wmean(x: pd.Series, w: pd.Series) -> float:
    xa = _to_float_array(x)
    wa = _to_float_array(w)
    mask = np.isfinite(xa) & np.isfinite(wa)
    xa, wa = xa[mask], wa[mask]
    if wa.size == 0 or np.sum(wa) == 0:
        return float("nan")
    return float(np.sum(wa * xa) / np.sum(wa))


def wsd(x: pd.Series, w: pd.Series) -> float:
    xa = _to_float_array(x)
    wa = _to_float_array(w)
    mask = np.isfinite(xa) & np.isfinite(wa)
    xa, wa = xa[mask], wa[mask]
    if wa.size == 0 or np.sum(wa) == 0:
        return float("nan")
    m = np.sum(wa * xa) / np.sum(wa)
    var = np.sum(wa * (xa - m) ** 2) / np.sum(wa)
    return float(np.sqrt(var))


def wcov(x: pd.Series, y: pd.Series, w: pd.Series) -> float:
    xa = _to_float_array(x)
    ya = _to_float_array(y)
    wa = _to_float_array(w)
    mask = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(wa)
    xa, ya, wa = xa[mask], ya[mask], wa[mask]
    if wa.size == 0 or np.sum(wa) == 0:
        return float("nan")
    mx = np.sum(wa * xa) / np.sum(wa)
    my = np.sum(wa * ya) / np.sum(wa)
    return float(np.sum(wa * (xa - mx) * (ya - my)) / np.sum(wa))


def wcorr(x: pd.Series, y: pd.Series, w: pd.Series) -> float:
    c = wcov(x, y, w)
    vx = wcov(x, x, w)
    vy = wcov(y, y, w)
    if not np.isfinite(c) or not np.isfinite(vx) or not np.isfinite(vy) or vx <= 0 or vy <= 0:
        return float("nan")
    return float(c / np.sqrt(vx * vy))


def wquantile(x: pd.Series, w: pd.Series, q: float) -> float:
    """
    Weighted quantile using the "smallest x such that cum_w >= q*sum_w" rule.
    """
    xa = _to_float_array(x)
    wa = _to_float_array(w)
    mask = np.isfinite(xa) & np.isfinite(wa)
    xa, wa = xa[mask], wa[mask]
    if wa.size == 0 or np.sum(wa) == 0:
        return float("nan")
    if q <= 0:
        return float(np.min(xa))
    if q >= 1:
        return float(np.max(xa))

    order = np.argsort(xa)
    xs = xa[order]
    ws = wa[order]
    cumw = np.cumsum(ws)
    cutoff = q * np.sum(ws)
    idx = int(np.searchsorted(cumw, cutoff, side="left"))
    idx = min(max(idx, 0), xs.size - 1)
    return float(xs[idx])


@dataclass(frozen=True)
class WlsSpec:
    y_col: str
    x_cols: Sequence[str]

    def params(self, df: pd.DataFrame, w: pd.Series) -> pd.Series:
        y = df[self.y_col]
        X = df[list(self.x_cols)]
        Xc = sm.add_constant(X, has_constant="add")
        fit = sm.WLS(y, Xc, weights=w).fit()
        return fit.params

    def fit_stats(self, df: pd.DataFrame, w: pd.Series) -> tuple[float, int]:
        y = df[self.y_col]
        X = df[list(self.x_cols)]
        Xc = sm.add_constant(X, has_constant="add")
        fit = sm.WLS(y, Xc, weights=w).fit()
        return float(fit.rsquared), int(fit.nobs)


@dataclass(frozen=True)
class LogitSpec:
    """
    Weighted logistic regression using statsmodels GLM Binomial with freq_weights.
    """
    y_col: str
    x_cols: Sequence[str]
    maxiter: int = 100

    def params(self, df: pd.DataFrame, w: pd.Series) -> pd.Series:
        y = df[self.y_col]
        X = df[list(self.x_cols)]
        Xc = sm.add_constant(X, has_constant="add")
        fit = sm.GLM(y, Xc, family=sm.families.Binomial(), freq_weights=w).fit(
            maxiter=self.maxiter, disp=0
        )
        return fit.params

def mean_estimator(y_col_or_cols: Union[str, Sequence[str]]):
    """
    Returns an estimator that produces a scalar mean.

    Accepts either a single column name, or a sequence of candidates (e.g., PV list).
    Using candidates is recommended with run_with_pvs(), since each PV loop subsets only that PV.
    """
    candidates = [y_col_or_cols] if isinstance(y_col_or_cols, str) else list(y_col_or_cols)

    def _est(df: pd.DataFrame, w: pd.Series) -> pd.Series:
        col = _pick_present_column(df, candidates)
        return pd.Series({"mean": wmean(df[col], w)})

    return _est

def proportion_estimator(y_col_or_cols: Union[str, Sequence[str]], cutoff: float, ge: bool = True):
    """
    Returns an estimator that produces a scalar proportion.
    - ge=True: P(y >= cutoff)
    - ge=False: P(y > cutoff)
    """
    candidates = [y_col_or_cols] if isinstance(y_col_or_cols, str) else list(y_col_or_cols)

    def _est(df: pd.DataFrame, w: pd.Series) -> pd.Series:
        col = _pick_present_column(df, candidates)
        y = pd.to_numeric(df[col], errors="coerce")
        ww = pd.to_numeric(w, errors="coerce")
        mask = np.isfinite(y.to_numpy(dtype=float)) & np.isfinite(ww.to_numpy(dtype=float))
        yv = y.to_numpy(dtype=float)[mask]
        wv = ww.to_numpy(dtype=float)[mask]
        if wv.size == 0 or np.sum(wv) == 0:
            return pd.Series({"proportion": float("nan")})

        ind = (yv >= cutoff).astype(float) if ge else (yv > cutoff).astype(float)
        p = float(np.sum(wv * ind) / np.sum(wv))
        return pd.Series({"proportion": p})

    return _est


def correlation(pv_cols: Sequence[str], other_col: str, name: str = "corr"):
    """
    Estimator: correlation between the current PV column and `other_col`.
    Returns Series({name: r}).
    """
    pv_candidates = list(pv_cols)

    def _est(df: pd.DataFrame, w: pd.Series) -> pd.Series:
        pv = _pick_present_column(df, pv_candidates)
        return pd.Series({name: wcorr(df[pv], df[other_col], w)})

    return _est


def percentiles(pv_cols: Sequence[str], probs: Sequence[float]):
    """
    Estimator: multiple weighted percentiles of the current PV.

    Returns Series with keys like p05, p10, p50, ...
    """
    pv_candidates = list(pv_cols)
    probs = list(probs)

    def _est(df: pd.DataFrame, w: pd.Series) -> pd.Series:
        pv = _pick_present_column(df, pv_candidates)
        out = {}
        for p in probs:
            key = f"p{int(round(p * 100)):02d}"
            out[key] = wquantile(df[pv], w, p)
        return pd.Series(out)

    return _est


def cumulative_benchmarks(pv_cols: Sequence[str], cutpoints: Sequence[float]):
    if cutpoints is None:
        raise ValueError("cutpoints must be provided (no universal default).")
    """
    Estimator: cumulative benchmark proportions P(PV >= cut) for each cutpoint.
    """
    pv_candidates = list(pv_cols)
    cuts = list(cutpoints)

    def _est(df: pd.DataFrame, w: pd.Series) -> pd.Series:
        pv = _pick_present_column(df, pv_candidates)
        y = pd.to_numeric(df[pv], errors="coerce").to_numpy(dtype=float)
        ww = pd.to_numeric(w, errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(ww)
        y, ww = y[mask], ww[mask]
        if ww.size == 0 or np.sum(ww) == 0:
            return pd.Series({f"ge_{c:g}": float("nan") for c in cuts})

        den = np.sum(ww)
        out = {}
        for c in cuts:
            out[f"ge_{c:g}"] = float(np.sum(ww * (y >= c)) / den)
        return pd.Series(out)

    return _est


def band_benchmarks(pv_cols: Sequence[str], cutpoints: Sequence[float]):
    if cutpoints is None:
        raise ValueError("cutpoints must be provided (no universal default).")
    """
    Estimator: non-cumulative benchmark band proportions.

    Bands are:
      <c1,
      [c1,c2),
      [c2,c3),
      ...
      >= last

    Keys are:
      lt_400, 400_474, 475_549, 550_624, ge_625   (for the default cutpoints)
    """
    pv_candidates = list(pv_cols)
    cuts = sorted(list(cutpoints))

    def _est(df: pd.DataFrame, w: pd.Series) -> pd.Series:
        pv = _pick_present_column(df, pv_candidates)
        y = pd.to_numeric(df[pv], errors="coerce").to_numpy(dtype=float)
        ww = pd.to_numeric(w, errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(ww)
        y, ww = y[mask], ww[mask]
        if ww.size == 0 or np.sum(ww) == 0:
            # construct the same keys we would otherwise return
            out = {f"lt_{cuts[0]:g}": float("nan")}
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                out[f"{lo:g}_{int(hi - 1):d}"] = float("nan")
            out[f"ge_{cuts[-1]:g}"] = float("nan")
            return pd.Series(out)

        den = np.sum(ww)
        out = {}

        out[f"lt_{cuts[0]:g}"] = float(np.sum(ww * (y < cuts[0])) / den)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            out[f"{lo:g}_{int(hi - 1):d}"] = float(np.sum(ww * ((y >= lo) & (y < hi))) / den)
        out[f"ge_{cuts[-1]:g}"] = float(np.sum(ww * (y >= cuts[-1])) / den)

        return pd.Series(out)

    return _est

