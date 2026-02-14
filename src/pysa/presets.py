from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Iterable, Union

import pandas as pd

from .replication import ReplicationDesign
from .core import (
    run_with_pvs,
    pysa_wls,
    pysa_correlation_with_pv,
    pysa_percentiles,
    pysa_cumulative_benchmarks,
    pysa_band_benchmarks,
)
from .results import PysaEstimate, PysaResult
from .estimators import mean_estimator, proportion_estimator
from .benchmarks import get_cutpoints


@dataclass(frozen=True)
class SurveySpec:
    pv_cols: Sequence[str]
    replication: ReplicationDesign

    # Study-specific defaults (optional)
    benchmark_cutpoints: Optional[Sequence[float]] = None
    benchmark_key: Optional[str] = None

    def mean(self, data: pd.DataFrame) -> PysaEstimate:
        # IMPORTANT: pass pv_cols (not pv_cols[0]) so PISA npv=10 works.
        est = mean_estimator(self.pv_cols)
        return run_with_pvs(
            data=data,
            pv_cols=self.pv_cols,
            replication=self.replication,
            estimator=est,
        )

    def proportion(self, data: pd.DataFrame, cutoff: float, ge: bool = True) -> PysaEstimate:
        # IMPORTANT: pass pv_cols (not pv_cols[0]) so PISA npv=10 works.
        est = proportion_estimator(self.pv_cols, cutoff=cutoff, ge=ge)
        return run_with_pvs(
            data=data,
            pv_cols=self.pv_cols,
            replication=self.replication,
            estimator=est,
        )

    def correlation_with_pv(self, data: pd.DataFrame, other_col: str) -> PysaEstimate:
        return pysa_correlation_with_pv(
            data=data,
            pv_cols=self.pv_cols,
            replication=self.replication,
            other_col=other_col,
        )

    def percentiles(self, data: pd.DataFrame, probs: Sequence[float]) -> PysaEstimate:
        return pysa_percentiles(
            data=data,
            pv_cols=self.pv_cols,
            replication=self.replication,
            probs=probs,
        )

    def _resolve_benchmark_cutpoints(
        self,
        cutpoints: Optional[Sequence[float]] = None,
        benchmark_key: Optional[str] = None,
    ) -> Sequence[float]:
        """
        Priority order:
        1) explicit cutpoints=...
        2) explicit benchmark_key=... (resolved via benchmarks.py)
        3) self.benchmark_cutpoints
        4) self.benchmark_key (resolved via benchmarks.py)
        else: error
        """
        if cutpoints is not None:
            return list(cutpoints)

        if benchmark_key is not None:
            return list(get_cutpoints(benchmark_key))

        if self.benchmark_cutpoints is not None:
            return list(self.benchmark_cutpoints)

        if self.benchmark_key is not None:
            return list(get_cutpoints(self.benchmark_key))

        raise ValueError(
            "No benchmark cutpoints available. Benchmarks are study-specific; "
            "provide cutpoints=..., or benchmark_key=..., or set SurveySpec.benchmark_cutpoints / benchmark_key."
        )

    def cumulative_benchmarks(
        self,
        data: pd.DataFrame,
        cutpoints: Optional[Sequence[float]] = None,
        benchmark_key: Optional[str] = None,
    ) -> PysaEstimate:
        cps = self._resolve_benchmark_cutpoints(cutpoints=cutpoints, benchmark_key=benchmark_key)
        return pysa_cumulative_benchmarks(
            data=data,
            pv_cols=self.pv_cols,
            replication=self.replication,
            cutpoints=cps,
        )

    def band_benchmarks(
        self,
        data: pd.DataFrame,
        cutpoints: Optional[Sequence[float]] = None,
        benchmark_key: Optional[str] = None,
    ) -> PysaEstimate:
        cps = self._resolve_benchmark_cutpoints(cutpoints=cutpoints, benchmark_key=benchmark_key)
        return pysa_band_benchmarks(
            data=data,
            pv_cols=self.pv_cols,
            replication=self.replication,
            cutpoints=cps,
        )

    def wls(
        self,
        data: pd.DataFrame,
        dep_root: str,
        num_vars: Sequence[str],
        cat_vars: Sequence[str],
        fixed_effects: Sequence[str] = (),
        country_var: str = "CNTRYID",
        country_values: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
        compute_standardized: bool = True,
    ) -> PysaResult:
        return pysa_wls(
            data=data,
            dep_root=dep_root,
            n_pv=len(self.pv_cols),
            pv_cols=self.pv_cols,
            replication=self.replication,
            num_vars=num_vars,
            cat_vars=cat_vars,
            fixed_effects=fixed_effects,
            country_var=country_var,
            country_values=country_values,
            compute_standardized=compute_standardized,
        )


def make_brr_fay_spec(
    pv_cols: Sequence[str],
    weight_col: str,
    replicate_cols: Sequence[str],
    fay_rho: float = 0.5,
    benchmark_cutpoints: Optional[Sequence[float]] = None,
    benchmark_key: Optional[str] = None,
) -> SurveySpec:
    rep = ReplicationDesign(
        method="brr_fay",
        weight_col=weight_col,
        replicate_cols=list(replicate_cols),
        fay_rho=fay_rho,
    )
    return SurveySpec(
        pv_cols=list(pv_cols),
        replication=rep,
        benchmark_cutpoints=benchmark_cutpoints,
        benchmark_key=benchmark_key,
    )


def make_iea_jk2_spec(
    pv_cols: Sequence[str],
    weight_col: str,
    zone_col: str,
    rep_col: str,
    n_zones: int,
    rep_values: tuple[int, int] = (1, 2),
    benchmark_cutpoints: Optional[Sequence[float]] = None,
    benchmark_key: Optional[str] = None,
) -> SurveySpec:
    rep = ReplicationDesign(
        method="iea_jk2",
        weight_col=weight_col,
        zone_col=zone_col,
        rep_col=rep_col,
        n_zones=n_zones,
        rep_values=rep_values,
    )
    return SurveySpec(
        pv_cols=list(pv_cols),
        replication=rep,
        benchmark_cutpoints=benchmark_cutpoints,
        benchmark_key=benchmark_key,
    )


def make_naep_jk_spec(
    pv_cols: Sequence[str],
    weight_col: str,
    replicate_cols: Sequence[str],
    benchmark_cutpoints: Optional[Sequence[float]] = None,
    benchmark_key: Optional[str] = None,
) -> SurveySpec:
    rep = ReplicationDesign(
        method="naep_jk",
        weight_col=weight_col,
        replicate_cols=list(replicate_cols),
    )
    return SurveySpec(
        pv_cols=list(pv_cols),
        replication=rep,
        benchmark_cutpoints=benchmark_cutpoints,
        benchmark_key=benchmark_key,
    )
