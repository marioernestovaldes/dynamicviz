import os
import sys

import pandas as pd
import numpy as np
from sklearn.datasets import make_s_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamicviz import boot, score


def _generate_data():
    X, y = make_s_curve(100, random_state=0)
    y = pd.DataFrame(y, columns=["label"])
    out = boot.generate(
        X, Y=y, method="tsne", B=2, save=False, random_seed=123, random_state=123
    )
    return X, out


def test_boot_generate_parallel_equivalence():
    X, y = make_s_curve(100, random_state=1)
    y = pd.DataFrame(y, columns=["label"])
    out_parallel = boot.generate(
        X,
        Y=y,
        method="tsne",
        B=2,
        save=False,
        random_seed=452,
        random_state=452,
        num_jobs=2,
    )
    out_single = boot.generate(
        X,
        Y=y,
        method="tsne",
        B=2,
        save=False,
        random_seed=452,
        random_state=452,
    )
    pd.testing.assert_frame_equal(out_parallel, out_single)


def test_variance_outputs():
    X, out = _generate_data()
    global_var = score.variance(out, method="global")
    random_var = score.variance(out, method="random", k=10)
    assert np.all(np.isfinite(global_var))
    assert np.all(np.isfinite(random_var))


def test_concordance_bounds():
    X, out = _generate_data()
    methods = [
        "spearman",
        "distortion",
        "jaccard",
        "mean_projection_error",
        "stretch",
    ]
    for m in methods:
        val = score.concordance(out, X, method=m, k=10, bootstrap_number=-1)
        arr = np.asarray(val)
        assert np.all((arr >= 0) & (arr <= 1))
