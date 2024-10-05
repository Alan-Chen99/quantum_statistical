import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2_contingency


def remove_nonzero_cols(matrix: NDArray):
    assert len(matrix.shape) == 2
    non_zero_columns = ~((matrix == 0).all(axis=0))
    filtered_matrix = matrix[:, non_zero_columns]
    return filtered_matrix


def chi2_contingency_maybezero(observed: NDArray, correction=True, lambda_=None):
    assert len(observed.shape) == 2
    non_zero_columns = ~((observed == 0).all(axis=0))
    filtered_matrix = observed[:, non_zero_columns]

    return chi2_contingency(filtered_matrix, correction, lambda_)
