import scipy
import numpy as np

def correlation_pearson(a, b):
    return scipy.stats.pearsonr(a, b)[0]
    # return np.corrcoef(a, b)[0, 1]


def distance_chebyshev(a: np.array, b: np.array, **kwargs) -> float:
    return scipy.spatial.distance.chebyshev(u=a, v=b)