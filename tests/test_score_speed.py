import timeit
import numpy as np
import pandas as pd
from sklearn.datasets import make_s_curve
from sklearn.metrics import pairwise_distances

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamicviz import boot, score

X, y = make_s_curve(200, random_state=0)
y = pd.DataFrame(y, columns=['label'])
DATA = boot.generate(X, Y=y, method="pca", B=2, save=False, random_seed=0, random_state=0)

def variance_fast(df):
    unique_ids = np.unique(df['original_index'])
    embeddings = [df[df['bootstrap_number']==b][['x1','x2']].values for b in sorted(df['bootstrap_number'].unique())]
    idxs_list = [df[df['bootstrap_number']==b]['original_index'].values for b in sorted(df['bootstrap_number'].unique())]
    n = len(unique_ids)
    dist_lists = [[[] for _ in range(n)] for _ in range(n)]
    for emb, idxs in zip(embeddings, idxs_list):
        dist = pairwise_distances(emb)
        for i, orig_i in enumerate(idxs):
            row_lists = dist_lists[orig_i]
            for j, orig_j in enumerate(idxs):
                row_lists[orig_j].append(dist[i, j])
    mean_pairwise_distance = np.mean([np.mean(d) for row in dist_lists for d in row])
    result = np.zeros(n)
    for i in range(n):
        variances = []
        row_lists = dist_lists[i]
        for j in range(n):
            arr = np.array(row_lists[j]) / mean_pairwise_distance
            variances.append(np.var(arr))
        result[i] = np.mean(variances)
    return result


def test_score_speed():
    baseline_time = timeit.timeit(lambda: score.variance(DATA, method='global'), number=1)
    optimized_time = timeit.timeit(lambda: variance_fast(DATA), number=1)

    baseline = score.variance(DATA, method='global')
    optimized = variance_fast(DATA)

    assert np.allclose(baseline, optimized)
    assert optimized_time <= baseline_time * 0.8
