"""
Score module
- Tools for computing variance score and stability score
- Tools for computing concordance score and ensemble concordance score
"""

# author: Eric David Sun <edsun@stanford.edu>
# (C) 2022
from __future__ import print_function, division

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import numpy as np

# from numba import njit, jit, prange
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings("ignore")


# Variance Score
def variance(
    df,
    method="global",
    k=20,
    X_orig=None,
    neighborhoods=None,
    normalize_pairwise_distance=False,
    include_original=True,
    return_times=False,
    n_jobs=-1,
):
    """
    Computes variances scores from the output dataframe (out) of boot.generate()

    Arguments:
        df = pandas dataframe, output of boot.generate()
        method = str, specifies the type of stability score to compute
            "global" - compute stability score across the entire dataset
            "random" - approximate global stability score by randomly selecting k "neighbors" for each observation
            "local" - compute stability over k-nearest neighbors (specify k, defaults to 20)
        k = int, when method is "local" or "random"; specifies the number of neighbors
        neighborhoods = array or list of size n, with elements labeling neighborhoods
        normalize_pairwise_distance = if True, then divide each set of d[i,j] by its mean before computing variance
        include_original = if True, include the original (bootstrap_number=-1) embedding in calculating scores
        return_times = True or False; if not False, returns a dictionary of run times broken down by components as the second output
        n_jobs = number of parallel jobs when computing pairwise distances (default -1 uses all cores)

    Returns:
        mean_variance_distances = numpy array with variance scores (mean variance in pairwise distance to neighborhood) for each observation
    """
    # keep track of run times
    rt_dict = {}

    # retrieve embeddings and bootstrap indices
    if include_original is True:
        embeddings = [
            np.array(df[df["bootstrap_number"] == b][["x1", "x2"]].values)
            for b in np.unique(df["bootstrap_number"])
        ]
        bootstrap_indices = [
            np.array(df[df["bootstrap_number"] == b]["original_index"].values)
            for b in np.unique(df["bootstrap_number"])
        ]
    else:
        embeddings = [
            np.array(df[df["bootstrap_number"] == b][["x1", "x2"]].values)
            for b in np.unique(df["bootstrap_number"])
            if b != -1
        ]
        bootstrap_indices = [
            np.array(df[df["bootstrap_number"] == b]["original_index"].values)
            for b in np.unique(df["bootstrap_number"])
            if b != -1
        ]

    # set up neighborhoods for variance score
    print("Setting up neighborhoods...")
    start_time = time.time()
    neighborhood_dict = get_neighborhood_dict(
        method,
        k,
        keys=np.unique(df["original_index"]),
        neighborhoods=neighborhoods,
        X_orig=X_orig,
    )
    neighborhood_time = time.time() - start_time
    rt_dict["neighborhood"] = neighborhood_time
    print("--- %s seconds ---" % neighborhood_time)

    # populate distance dict
    print("Populating distances...")
    start_time = time.time()
    dist_dict = populate_distance_dict(
        neighborhood_dict, embeddings, bootstrap_indices, n_jobs=n_jobs
    )
    dist_time = time.time() - start_time
    rt_dict["distances"] = dist_time
    print("--- %s seconds ---" % dist_time)

    # compute mean pairwise distance for normalization
    print("Computing mean pairwise distance for normalization...")
    start_time = time.time()
    mean_pairwise_distance = compute_mean_distance(
        dist_dict, normalize_pairwise_distance
    )
    norm_time = time.time() - start_time
    rt_dict["normalization"] = norm_time
    print("--- %s seconds ---" % norm_time)

    # compute variances
    print("Computing variance scores...")
    start_time = time.time()
    mean_variance_distances = compute_mean_variance_distance(
        dist_dict, normalize_pairwise_distance, mean_pairwise_distance
    )
    var_time = time.time() - start_time
    rt_dict["variance"] = var_time
    print("--- %s seconds ---" % var_time)

    if return_times is False:
        return mean_variance_distances
    else:
        return (mean_variance_distances, rt_dict)


def stability_from_variance(mean_variance_distances, alpha):
    """
    For alpha and mean_variance_distances, computes stability scores

    Arguments:
        mean_variance_distances = list or array of variance scores (output of variance())
        alpha = float > 0, is the exponential paramter for stability score formula: stability = 1/(1+variance)^alpha
            defaults to alpha=1.0

    Returns:
        stability_scores = numpy array with stability score for each observation
    """
    # compute stability score
    print("Computing stability score with alpha=" + str(alpha) + " ...")
    start_time = time.time()
    stability_scores = 1 / (1 + mean_variance_distances) ** alpha
    print("--- %s seconds ---" % (time.time() - start_time))

    return stability_scores


# Stability score
def stability(
    df,
    method="global",
    alpha=1.0,
    k=20,
    X_orig=None,
    neighborhoods=None,
    normalize_pairwise_distance=False,
    include_original=True,
    return_times=False,
):
    """
    Computes stability scores from the output dataframe (out) of boot.generate()

    Arguments:
        alpha = float > 0, is the exponential paramter for stability score formula: stability = 1/(1+variance)^alpha
            defaults to alpha=1.0
        See variance() for more details

    Returns:
        stability_scores = numpy array with stability score for each observation
    """
    # check alpha > 0
    if alpha <= 0:
        raise Exception("alpha must be >= 0")

    if return_times is False:
        mean_variance_distances = variance(
            df,
            method=method,
            k=k,
            X_orig=X_orig,
            neighborhoods=neighborhoods,
            normalize_pairwise_distance=normalize_pairwise_distance,
            include_original=include_original,
            return_times=return_times,
        )
    else:
        mean_variance_distances, rt_dict = variance(
            df,
            method=method,
            k=k,
            X_orig=X_orig,
            neighborhoods=neighborhoods,
            normalize_pairwise_distance=normalize_pairwise_distance,
            include_original=include_original,
            return_times=return_times,
        )

    # compute stability score
    stability_scores = stability_from_variance(mean_variance_distances, alpha)

    if return_times is False:
        return stability_scores
    else:
        return (stability_scores, rt_dict)


# @njit(parallel=True)
def get_neighborhood_dict(method, k, keys, neighborhoods=None, X_orig=None):
    """
    Returns a neighborhood dictionary where keys are observation indices

    Arguments:
        method = string specifier for the method for constructing neighborhoods ("global", "random", "local")
            to specify a predefined/custom neighborhood, use the neighborhoods argument
        k = integer specfiying size of neighborhood for "local" and "random" methods
        keys = list/array of the key names (integers) to use in constructing the dictionary
        neighborhoods = predefined neighborhoods [list of strings where same strings indicate observations in the same neighborhood]
        X_orig = original X used to computing the bootstraps; used to determine local kNN relations

    Returns:
        neighborhood_dict = dictionary with keys corresponding to observation indices and lists of their neighborhoods as values
    """
    neighborhood_dict = {int(key): [] for key in keys}

    if neighborhoods is None:

        if method == "global":
            for key in list(neighborhood_dict.keys()):
                neighborhood_dict[key] = [
                    i for i in range(len(neighborhood_dict.keys()))
                ]

        elif method == "random":
            for key in list(neighborhood_dict.keys()):
                neighborhood_dict[key] = [
                    int(i)
                    for i in np.random.randint(0, len(neighborhood_dict.keys()), k)
                ]

        elif method == "local":
            if X_orig is None:
                raise Exception("Need to specify X_orig to compute nearest neighbors")
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_orig)
            distances_local, indices_local = nbrs.kneighbors(X_orig)
            for key in list(neighborhood_dict.keys()):
                neighborhood_dict[key] = [
                    int(i) for i in indices_local[int(key), 1 : k + 1]
                ]

        else:
            raise Exception(
                "Need to specify either method ('global', 'local') or neighborhood"
            )

    else:
        for key in list(neighborhood_dict.keys()):
            label = neighborhoods[int(key)]
            neighborhood_dict[key] = [
                i for i in range(len(neighborhoods)) if neighborhoods[i] == label
            ]

    return neighborhood_dict


# @njit(parallel=True)
def populate_distance_dict(neighborhood_dict, embeddings, bootstrap_indices, n_jobs=-1):
    """
    Returns dictionary with pairwise dictionaries for all observations[i][j]

    Arguments:
        neighborhood_dict = dictionary with keys corresponding to observation indices and lists of their neighborhoods as values
            this is the output of get_neighborhood_dict()
        embeddings = list of nx2 DR visualization arrays [see outputs of boot.generate()]
        bootstrap_indices = list of arrays, where each array is the bootstrap indices of that visualization [see outputs of boot.generate()]

    Returns:
        dist_dict = two-level dictionary with first level corresponding to observation i and second level corresponding to observation j
                    the value corresponds to a list of Euclidean distances between observation i and j across all co-occurrences in the bootstrap visualizations
    """
    dist_dict = {
        str(key1): {str(key2): [] for key2 in neighborhood_dict[key1]}
        for key1 in neighborhood_dict.keys()
    }

    for emb, boot_idxs in zip(embeddings, bootstrap_indices):
        for i_idx, orig_i in enumerate(boot_idxs):
            key1 = str(orig_i)
            neighbor_js = neighborhood_dict[orig_i]

            pairs = []
            for nj in neighbor_js:
                indices_j = np.where(boot_idxs == nj)[0]
                for pos in indices_j:
                    pairs.append((pos, str(nj)))

            if pairs:
                sub_emb = emb[[i_idx] + [p for p, _ in pairs]]
                dists = pairwise_distances(sub_emb, n_jobs=n_jobs)[0, 1:]
                for (pos, key2), dist in zip(pairs, dists):
                    if pos == i_idx:
                        dist = 0.0
                    dist_dict[key1][key2].append(dist)

    for key1 in dist_dict.keys():
        for key2 in dist_dict[key1].keys():
            dist_dict[key1][key2] = np.asarray(dist_dict[key1][key2], dtype=float)

    return dist_dict


# @njit(parallel=True)
def compute_mean_distance(dist_dict, normalize_pairwise_distance=False):
    """
    Computes mean pairwise distance across all (i,j)

    Arguments:
        dist_dict = output of populate_distance_dict()
        normalize_pairwise_distance = boolean; whether to normalize the distances between i and j by their mean

    Returns:
        mean_pairwise_distance = the mean of all distances between any i and any j; used to normalize/scale the variance score
    """
    arrays = [
        dist_dict[key1][key2].astype(float)
        for key1 in dist_dict.keys()
        for key2 in dist_dict[key1].keys()
    ]

    if not arrays:
        return np.nan

    maxlen = max(len(a) for a in arrays)
    distances = np.full((len(arrays), maxlen), np.nan, dtype=float)
    for idx, a in enumerate(arrays):
        distances[idx, : len(a)] = a

    if normalize_pairwise_distance is True:
        distances /= np.nanmean(distances, axis=1, keepdims=True)

    mean_pairwise_distance = np.nanmean(np.nanmean(distances, axis=1))

    return mean_pairwise_distance


# @njit(parallel=True)
def compute_mean_variance_distance(
    dist_dict, normalize_pairwise_distance=False, mean_pairwise_distance=1.0
):
    """
    For each (i,j) compute the variance across all distances.
    Then for each i, average across all var(i,j)

    Arguments:
        dist_dict = output of populate_distance_dict()
        normalize_pairwise_distance = boolean; whether to normalize the distances between i and j by their mean
        mean_pairwise_distance = float, output of compute_mean_distance()

    Returns:
        mean_variance_distances = list of variance scores [one for each observation]
    """
    mean_variance_distances = np.ones(len(dist_dict.keys())) * np.inf

    for key1 in dist_dict.keys():
        arrays = [
            dist_dict[key1][key2].astype(float) for key2 in dist_dict[key1].keys()
        ]
        if not arrays:
            continue

        maxlen = max(len(a) for a in arrays)
        distances = np.full((len(arrays), maxlen), np.nan, dtype=float)
        for idx, a in enumerate(arrays):
            distances[idx, : len(a)] = a

        if normalize_pairwise_distance is True:
            distances /= np.nanmean(distances, axis=1, keepdims=True)

        variances = np.nanvar(distances / mean_pairwise_distance, axis=1)
        mean_variance_distances[int(key1)] = np.nanmean(variances)

    return mean_variance_distances


### CONCORDANCE SCORES -- see our publication for mathematical details


def get_jaccard(X_orig, X_red, k, precomputed=[False, False]):
    """Return the Jaccard coefficient at ``k`` for each observation.

    Parameters
    ----------
    X_orig : array-like or ndarray
        Original data or a precomputed pairwise distance matrix when
        ``precomputed[0]`` is ``True``.
    X_red : array-like or ndarray
        Reduced data or a precomputed pairwise distance matrix when
        ``precomputed[1]`` is ``True``.
    k : int
        Number of nearest neighbours to consider **excluding** the point
        itself.
    precomputed : list of bool, default ``[False, False]``
        Flags indicating whether ``X_orig`` and ``X_red`` are already distance
        matrices.

    Returns
    -------
    numpy.ndarray
        Array of Jaccard coefficients for each observation.
    """

    if not precomputed[0]:
        dist_orig = pairwise_distances(X_orig, n_jobs=-1)
    else:
        dist_orig = np.asarray(X_orig)

    if not precomputed[1]:
        dist_red = pairwise_distances(X_red, n_jobs=-1)
    else:
        dist_red = np.asarray(X_red)

    indices_orig = np.argsort(dist_orig, axis=1)[:, 1 : k + 1]
    indices_red = np.argsort(dist_red, axis=1)[:, 1 : k + 1]

    jaccards = []
    for neigh_o, neigh_r in zip(indices_orig, indices_red):
        inter = len(np.intersect1d(neigh_o, neigh_r))
        union = 2 * k - inter
        jaccards.append(inter / union)

    return np.asarray(jaccards, dtype=float)


def get_distortion(X_orig, X_red, k, precomputed=[False, False]):
    """Return the distortion at ``k`` for each observation.

    Distortion is defined as

    ``abs(log((D_furthest/D_nearest)_orig / (D_furthest/D_nearest)_red))``.
    Values are normalised to ``[0, 1]`` and inverted so that ``1`` is best.

    Parameters
    ----------
    X_orig : array-like or ndarray
        Original data or a precomputed pairwise distance matrix when
        ``precomputed[0]`` is ``True``.
    X_red : array-like or ndarray
        Reduced data or a precomputed pairwise distance matrix when
        ``precomputed[1]`` is ``True``.
    k : int
        Number of nearest neighbours to consider **excluding** the point
        itself.
    precomputed : list of bool, default ``[False, False]``
        Flags indicating whether ``X_orig`` and ``X_red`` are already distance
        matrices.

    Returns
    -------
    numpy.ndarray
        Array of distortion values for each observation.
    """

    if not precomputed[0]:
        dist_orig = pairwise_distances(X_orig, n_jobs=-1)
    else:
        dist_orig = np.asarray(X_orig)

    if not precomputed[1]:
        dist_red = pairwise_distances(X_red, n_jobs=-1)
    else:
        dist_red = np.asarray(X_red)

    sorted_orig = np.sort(dist_orig, axis=1)
    sorted_red = np.sort(dist_red, axis=1)

    eps = np.finfo(float).eps
    orig_ratio = sorted_orig[:, k] / np.maximum(sorted_orig[:, 1], eps)
    red_ratio = sorted_red[:, k] / np.maximum(sorted_red[:, 1], eps)

    distortions = np.abs(np.log(orig_ratio / red_ratio))
    distortions = distortions / np.max(distortions)
    distortions = 1 - distortions

    return distortions


def get_mean_projection_error(X_orig, X_red):
    """
    Computes mean projection error (MPE) modified from aggregated projection error by Martins et al., 2014

    MPE_i = MEAN_j { ABS[ D[i,j]_orig / max(D[i,j]_orig) - D[i,j]_red / max(D[i,j]_red) ] }

    Normalized to [0,1] and then reframed so 1 is best
    """
    orig_distance = pairwise_distances(X_orig)
    red_distance = pairwise_distances(X_red)

    # normalize distances
    eps = np.finfo(float).eps
    for i in range(orig_distance.shape[0]):
        max_orig = np.maximum(np.max(orig_distance[i, :]), eps)
        max_red = np.maximum(np.max(red_distance[i, :]), eps)
        orig_distance[i, :] = orig_distance[i, :] / max_orig
        red_distance[i, :] = red_distance[i, :] / max_red

    # compute projection errors and then MPE
    projection_errors = np.abs(orig_distance - red_distance)
    MPEs = np.mean(projection_errors, axis=1)
    MPEs = 1 - MPEs / np.max(MPEs)
    return MPEs


def get_stretch(X_orig, X_red):
    """
    Implemented according to Aupetit, 2007:

    stretch_i = [ u_i - min_k {u_k} ] / [ max_k {u_k} - min_k {u_k} ]

    u_i = SUM_j D_ij^+  , D_ij^+ = max {-(D_orig_ij - D_red_ij), 0 }  -- D here being Euclidean distance matrix

    stretchs reframed so 1 is best
    """
    orig_distance = pairwise_distances(X_orig)
    red_distance = pairwise_distances(X_red)

    # normalize distances
    for i in range(orig_distance.shape[0]):
        orig_distance[i, :] = orig_distance[i, :] / np.linalg.norm(orig_distance[i, :])
        red_distance[i, :] = red_distance[i, :] / np.linalg.norm(red_distance[i, :])

    D_neg = orig_distance - red_distance
    D_neg[D_neg > 0] = 0
    D_neg = -D_neg

    U = np.sum(D_neg, axis=1)
    stretchs = (U - np.min(U)) / (np.max(U) - np.min(U))
    stretchs = 1 - stretchs

    return stretchs


def concordance(df, X_orig, method, k=None, bootstrap_number=-1):
    """
    Computes concordance scores between the projections in df and X_orig

    Arguments:
        df = pandas dataframe: output of boot.generate()
        X_orig = nxp numpy array that is the original data from which df was generated
        method = str: 'spearman', 'jaccard', 'distortion',
                 'mean_projection_error', 'stretch'
        k = int, neighborhood size to consider (jaccard, distortion, projection_precision_score, spearman, stretch)
        bootstrap_number = int, index of bootstrap to compute metrics for; defaults to -1 which is the original/unbootstrapped projection

    Returns:
        metrics = numpy array with quality score for each row of df (according to the method specified) [0 is bad, 1 is good]
    """
    # retrieve embeddings
    X_red = df[df["bootstrap_number"] == bootstrap_number][["x1", "x2"]].values

    # shuffle X_orig to matching format
    boot_idxs = df[df["bootstrap_number"] == bootstrap_number]["original_index"].values
    X_orig = X_orig[boot_idxs, :]

    assert (
        X_orig.shape[0] == X_red.shape[0]
    ), "Error: number of observations are not consistent"

    # set k to a globally relevant value if None
    if k is None:
        k = round(X_orig.shape[0] / 2 - 1)
    if k < 5:
        raise Exception(
            "k needs to be >= 5 or number of observations in X is too small"
        )

    if method == "spearman":
        orig_distance = pairwise_distances(X_orig)
        red_distance = pairwise_distances(X_red)
        metrics = [
            spearmanr(orig_distance[i, :], red_distance[i, :])[0]
            for i in range(red_distance.shape[0])
        ]
    elif method == "jaccard":
        metrics = get_jaccard(X_orig, X_red, k)
    elif method == "distortion":
        metrics = get_distortion(X_orig, X_red, k)
    elif method == "mean_projection_error":
        metrics = get_mean_projection_error(X_orig, X_red)
    elif method == "stretch":
        metrics = get_stretch(X_orig, X_red)
    else:
        raise Exception("method not recognized")

    return metrics


def ensemble_concordance(
    df, X_orig, methods=None, k=None, bootstrap_number=-1, verbose=True
):
    """
    Compute ensemble concordance via spectral meta-weights.

    Arguments:
        df = pandas dataframe: output of boot.generate()
        X_orig = nxp numpy array that is the original data from which df was generated
        methods = list of strings specifying the metrics to use
            Defaults to all the available metrics (except for precision and recall)
        k = int, neighborhood size to consider (jaccard, distortion, projection_precision_score, precision, recall)
        bootstrap_number = int, index of bootstrap to compute metrics for; defaults to -1 which is the original/unbootstrapped projection
        verbose = True or False, whether to return warnings about negative spectral weights

    Returns:
        ensemble_metric = numpy array of ensemble concordance scores
        pointwise_metrics_list = list of concordance score arrays corresponding to pointwise_metrics_labels
    """
    if methods is None:
        methods = [
            "spearman",
            "jaccard",
            "distortion",
            "mean_projection_error",
            "stretch",
        ]

    # compute individual metrics
    pointwise_metrics_list = []

    for metric in tqdm(methods):
        m = concordance(
            df, X_orig, method=metric, k=k, bootstrap_number=bootstrap_number
        )
        pointwise_metrics_list.append(np.array(m))

    # compute correlation matrix and eigendecomposition
    mat = np.corrcoef(pointwise_metrics_list)
    w, v = np.linalg.eig(mat)
    pc1_score = v[:, 0]

    # check pc1 for one sign only
    summed_signs = np.abs(np.sum(np.sign(pc1_score)))
    if verbose is True:
        if summed_signs == len(pc1_score):
            print("PC1 is same signed")
        else:
            print("Warning: PC1 is mixed signed")

    # compute meta-uncertainty
    weighted_metrics_list = [
        pointwise_metrics_list[i] * np.abs(pc1_score)[i]
        for i in range(len(pointwise_metrics_list))
    ]
    ensemble_metric = np.sum(weighted_metrics_list, axis=0) / np.sum(np.abs(pc1_score))

    return (ensemble_metric, pointwise_metrics_list)
