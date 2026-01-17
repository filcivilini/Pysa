from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _to_float_array(x: pd.Series) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)


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


def mean_estimator(y_col: str):
    def _est(df: pd.DataFrame, w: pd.Series) -> pd.Series:
        return pd.Series({"mean": wmean(df[y_col], w)})
    return _est


def proportion_estimator(y_col: str, cutoff: float, ge: bool = True):
    def _est(df: pd.DataFrame, w: pd.Series) -> pd.Series:
        y = pd.to_numeric(df[y_col], errors="coerce")
        ww = pd.to_numeric(w, errors="coerce")
        mask = np.isfinite(y.to_numpy(dtype=float)) & np.isfinite(ww.to_numpy(dtype=float))
        yv = y.to_numpy(dtype=float)[mask]
        wv = ww.to_numpy(dtype=float)[mask]
        if wv.size == 0 or np.sum(wv) == 0:
            return pd.Series({"proportion": float("nan")})
        if ge:
            ind = (yv >= cutoff).astype(float)
        else:
            ind = (yv > cutoff).astype(float)
        p = float(np.sum(wv * ind) / np.sum(wv))
        return pd.Series({"proportion": p})
    return _est
