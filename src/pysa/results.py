from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class PysaEstimate:
    estimate: pd.Series
    se: pd.Series

    def to_dataframe(self) -> pd.DataFrame:
        out = pd.DataFrame({"estimate": self.estimate, "se": self.se})
        out["t"] = out["estimate"] / out["se"]
        return out


@dataclass
class PysaResult:
    coef: pd.Series
    se: pd.Series
    std_coef: pd.Series
    std_se: pd.Series
    r2: float
    n_obs: int
    dep_root: str

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "Variable": self.coef.index,
                "coef": self.coef.values,
                "se": self.se.reindex(self.coef.index).values,
                "std_coef": self.std_coef.reindex(self.coef.index).values,
                "std_se": self.std_se.reindex(self.coef.index).values,
            }
        )
        df["t"] = df["coef"] / df["se"]
        df["std_t"] = df["std_coef"] / df["std_se"]
        return df
