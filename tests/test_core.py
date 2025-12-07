import numpy as np
import pandas as pd

from pysa import pysa_wls


def test_pysa_wls_smoke_minimal():
    n = 12

    df = pd.DataFrame({
        "PV1MATH": np.random.randn(n),

        "W_FSTUWT": np.ones(n),

        "W_FSTURWT1": np.ones(n),
        "W_FSTURWT2": np.ones(n),

        "ESCS": np.random.randn(n),

        "GENDER": np.random.choice([1, 2], size=n),

        "CNTRYID": [380] * n,
    })

    res = pysa_wls(
        data=df,
        dep_root="MATH",
        n_pv=1,
        num_vars=["ESCS"],
        cat_vars=["GENDER"],
        fixed_effects=[],
        weight_col="W_FSTUWT",
        replicate_weight_prefix="W_FSTURWT",
        n_replicates=2,
        fay_rho=0.5,
        country_var="CNTRYID",
        country_values=380,
    )

    assert "ESCS" in res.coef.index
    assert res.n_obs > 0
