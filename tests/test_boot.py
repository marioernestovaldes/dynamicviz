import numpy as np
from dynamicviz import boot


def test_dimensionality_reduction_pca():
    X = np.random.RandomState(0).randn(30, 5)
    result = boot.dimensionality_reduction(X, 'pca')
    assert result.shape == (30, 2)
