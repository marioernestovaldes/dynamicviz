import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_s_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamicviz import boot, score
from sklearn.metrics import pairwise_distances


# Baseline implementations copied from the previous version of score.py


def compute_mean_distance_old(dist_dict, normalize_pairwise_distance=False):
    all_distances = []
    for key1 in dist_dict.keys():
        for key2 in dist_dict[key1].keys():
            distances = dist_dict[key1][key2]
            if normalize_pairwise_distance:
                distances = distances / np.nanmean(distances)
            all_distances.append(np.nanmean(distances))
    return np.nanmean(all_distances)


def compute_mean_variance_distance_old(
    dist_dict, normalize_pairwise_distance=False, mean_pairwise_distance=1.0
):
    mean_variance_distances = np.ones(len(dist_dict.keys())) * np.inf
    for key1 in dist_dict.keys():
        variances = []
        for key2 in dist_dict[key1].keys():
            distances = dist_dict[key1][key2]
            if normalize_pairwise_distance:
                distances = distances / np.nanmean(distances)
            variances.append(np.nanvar(distances / mean_pairwise_distance))
        mean_variance_distances[int(key1)] = np.nanmean(variances)
    return mean_variance_distances


def populate_distance_dict_old(neigh, embeddings, boot_idxs):
    dist_dict = {
        str(key1): {str(key2): [] for key2 in neigh[key1]} for key1 in neigh.keys()
    }

    for emb, idxs in zip(embeddings, boot_idxs):
        dist_mat = pairwise_distances(emb)
        for i, orig_i in enumerate(idxs):
            key1 = str(orig_i)
            for nj in neigh[orig_i]:
                key2 = str(nj)
                js = np.where(idxs == nj)[0]
                if js.size:
                    dist_dict[key1][key2].extend(dist_mat[i, js])

    for key1 in dist_dict:
        for key2 in dist_dict[key1]:
            dist_dict[key1][key2] = np.asarray(dist_dict[key1][key2], dtype=float)

    return dist_dict


def test_distance_functions_equivalence():
    X, y = make_s_curve(30, random_state=0)
    y = pd.DataFrame(y, columns=["label"])
    data = boot.generate(
        X, Y=y, method="pca", B=2, save=False, random_seed=0, random_state=0
    )

    embeddings = [
        data[data["bootstrap_number"] == b][["x1", "x2"]].values
        for b in np.unique(data["bootstrap_number"])
    ]
    boot_idxs = [
        data[data["bootstrap_number"] == b]["original_index"].values
        for b in np.unique(data["bootstrap_number"])
    ]
    neigh = score.get_neighborhood_dict(
        "global", k=5, keys=np.unique(data["original_index"])
    )
    dist_dict = score.populate_distance_dict(neigh, embeddings, boot_idxs)

    mean_new = score.compute_mean_distance(dist_dict, normalize_pairwise_distance=True)
    mean_old = compute_mean_distance_old(dist_dict, normalize_pairwise_distance=True)
    assert np.allclose(mean_new, mean_old, atol=1e-8)

    var_new = score.compute_mean_variance_distance(
        dist_dict, normalize_pairwise_distance=True, mean_pairwise_distance=mean_new
    )
    var_old = compute_mean_variance_distance_old(
        dist_dict, normalize_pairwise_distance=True, mean_pairwise_distance=mean_old
    )
    assert np.allclose(var_new, var_old, atol=1e-8)


def test_populate_distance_dict_equivalence():
    X, y = make_s_curve(30, random_state=1)
    y = pd.DataFrame(y, columns=["label"])
    data = boot.generate(
        X, Y=y, method="pca", B=2, save=False, random_seed=1, random_state=0
    )

    embeddings = [
        data[data["bootstrap_number"] == b][["x1", "x2"]].values
        for b in np.unique(data["bootstrap_number"])
    ]
    boot_idxs = [
        data[data["bootstrap_number"] == b]["original_index"].values
        for b in np.unique(data["bootstrap_number"])
    ]
    neigh = score.get_neighborhood_dict(
        "global", k=5, keys=np.unique(data["original_index"])
    )

    dist_new = score.populate_distance_dict(neigh, embeddings, boot_idxs)
    dist_old = populate_distance_dict_old(neigh, embeddings, boot_idxs)

    for key1 in dist_old:
        for key2 in dist_old[key1]:
            assert np.allclose(dist_new[key1][key2], dist_old[key1][key2])
