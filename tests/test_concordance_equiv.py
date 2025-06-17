import os
import sys

import numpy as np
from sklearn.datasets import make_s_curve
from sklearn.metrics import pairwise_distances

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamicviz import score, boot


def jaccard_reference(X_orig, X_red, k):
    dist_orig = pairwise_distances(X_orig)
    dist_red = pairwise_distances(X_red)
    indices_orig = np.argsort(dist_orig, axis=1)[:, 1 : k + 1]
    indices_red = np.argsort(dist_red, axis=1)[:, 1 : k + 1]
    result = []
    for neigh_o, neigh_r in zip(indices_orig, indices_red):
        inter = len(np.intersect1d(neigh_o, neigh_r))
        union = 2 * k - inter
        result.append(inter / union)
    return np.array(result)


def distortion_reference(X_orig, X_red, k):
    dist_orig = pairwise_distances(X_orig)
    dist_red = pairwise_distances(X_red)
    sorted_orig = np.sort(dist_orig, axis=1)
    sorted_red = np.sort(dist_red, axis=1)
    orig_ratio = sorted_orig[:, k] / sorted_orig[:, 1]
    red_ratio = sorted_red[:, k] / sorted_red[:, 1]
    distortions = np.abs(np.log(orig_ratio / red_ratio))
    distortions = distortions / np.max(distortions)
    return 1 - distortions


def test_jaccard_and_distortion_equivalence():
    X, _ = make_s_curve(50, random_state=0)
    data = boot.generate(
        X, method="pca", B=1, save=False, random_seed=0, random_state=0
    )
    X_red = data[data["bootstrap_number"] == -1][["x1", "x2"]].values

    k = 5
    ref_jaccard = jaccard_reference(X, X_red, k)
    ref_distortion = distortion_reference(X, X_red, k)

    jaccard = score.get_jaccard(X, X_red, k)
    distortion = score.get_distortion(X, X_red, k)

    assert np.allclose(jaccard, ref_jaccard, atol=1e-8)
    assert np.allclose(distortion, ref_distortion, atol=1e-8)
