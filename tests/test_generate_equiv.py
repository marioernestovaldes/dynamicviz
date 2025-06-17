import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_s_curve
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamicviz import boot


def generate_reference(
    X,
    method,
    Y=None,
    B=0,
    sigma_noise=None,
    no_bootstrap=False,
    random_seed=None,
    num_jobs=None,
    use_n_pcs=False,
    subsample=False,
    **kwargs,
):
    """Replicates the original generate behavior using sequential concatenation."""
    output = pd.DataFrame()

    original_embedding = boot.dimensionality_reduction(X, method, **kwargs)
    points0 = np.hstack(
        (original_embedding, np.zeros((original_embedding.shape[0], 1)))
    )

    output["x1"] = points0[:, 0] - np.mean(points0[:, 0])
    output["x2"] = points0[:, 1] - np.mean(points0[:, 1])
    output["original_index"] = np.arange(len(points0[:, 0]))
    output["bootstrap_number"] = -1

    if isinstance(Y, pd.DataFrame):
        for col in Y.columns:
            output[col] = Y[col].values

    if B > 0:
        bootstrap_embedding_list, bootstrap_indices_list = boot.bootstrap(
            X,
            method,
            B,
            sigma_noise,
            no_bootstrap,
            random_seed,
            num_jobs,
            use_n_pcs,
            subsample,
            **kwargs,
        )
    else:
        bootstrap_embedding_list, bootstrap_indices_list = [], []

    for i in range(len(bootstrap_embedding_list)):
        new_df = pd.DataFrame()
        points = bootstrap_embedding_list[i]
        points = np.hstack((points, np.zeros((points.shape[0], 1))))
        boot_idxs = bootstrap_indices_list[i]

        ref_points = points0[boot_idxs, :]
        points[:, 0] = points[:, 0] - np.mean(points[:, 0])
        points[:, 1] = points[:, 1] - np.mean(points[:, 1])
        r = Rotation.align_vectors(ref_points, points)[0]
        rpoints = r.apply(points)

        new_df["x1"] = rpoints[:, 0]
        new_df["x2"] = rpoints[:, 1]
        new_df["original_index"] = boot_idxs
        new_df["bootstrap_number"] = i

        if isinstance(Y, pd.DataFrame):
            for col in Y.columns:
                new_df[col] = Y[col].values[boot_idxs]

        output = pd.concat([output, new_df], axis=0)

    return output


def test_generate_output_equivalence():
    X, y = make_s_curve(30, random_state=0)
    y_df = pd.DataFrame(y, columns=["label"])

    ref = generate_reference(X, "pca", Y=y_df, B=2, random_seed=0, random_state=0)
    new = boot.generate(
        X, Y=y_df, method="pca", B=2, save=False, random_seed=0, random_state=0
    )

    pd.testing.assert_frame_equal(
        ref.reset_index(drop=True), new.reset_index(drop=True)
    )
