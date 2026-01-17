from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Literal

import numpy as np
import pandas as pd


RepMethod = Literal["brr_fay", "iea_jk2", "naep_jk"]


@dataclass(frozen=True)
class ReplicationDesign:
    method: RepMethod
    weight_col: str

    replicate_cols: Optional[Sequence[str]] = None

    zone_col: Optional[str] = None
    rep_col: Optional[str] = None
    n_zones: Optional[int] = None
    rep_values: tuple[int, int] = (1, 2)

    fay_rho: float = 0.5
    replicate_scale_override: Optional[float] = None

    def validate(self, df: pd.DataFrame) -> None:
        if self.weight_col not in df.columns:
            raise ValueError(f"Weight column '{self.weight_col}' not found.")

        if self.replicate_cols is not None:
            missing = [c for c in self.replicate_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing replicate weight columns: {missing}")
            return

        if self.zone_col is None or self.rep_col is None or self.n_zones is None:
            raise ValueError(
                "ReplicationDesign requires either replicate_cols or (zone_col, rep_col, n_zones)."
            )

        missing = [c for c in (self.zone_col, self.rep_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required JK columns: {missing}")

    def iter_replicate_weights(self, df: pd.DataFrame, base_w: pd.Series) -> Iterator[pd.Series]:
        if self.replicate_cols is not None:
            for c in self.replicate_cols:
                yield df[c]
            return

        zone = pd.to_numeric(df[self.zone_col], errors="coerce").to_numpy(dtype=float)
        rep = pd.to_numeric(df[self.rep_col], errors="coerce").to_numpy(dtype=float)
        bw = pd.to_numeric(base_w, errors="coerce").to_numpy(dtype=float)

        r1, r2 = self.rep_values
        for h in range(1, int(self.n_zones) + 1):
            for rv in (r1, r2):
                factor = np.where(zone == h, np.where(rep == rv, 2.0, 0.0), 1.0)
                yield pd.Series(bw * factor, index=df.index, name=f"rep_z{h}_r{rv}")

    def replicate_scale(self, n_reps: int) -> float:
        if self.replicate_scale_override is not None:
            return float(self.replicate_scale_override)

        if self.method == "brr_fay":
            k = float(self.fay_rho)
            return 1.0 / (n_reps * (1.0 - k) ** 2)

        if self.method == "iea_jk2":
            return 0.5

        if self.method == "naep_jk":
            return 1.0

        raise ValueError(f"Unknown replication method: {self.method}")


def sampling_variance(theta_full: pd.Series, thetas_rep: pd.DataFrame, scale: float) -> pd.Series:
    diff = thetas_rep.sub(theta_full, axis=1)
    return scale * (diff**2).sum(axis=0)
