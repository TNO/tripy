import numpy as np
import pytest

from tripy.utils import correlation_function, correlation_matrix, inv_cov_vec_1D


def test_correlation_function_wrong_d_input():
    with pytest.raises(ValueError):
        correlation_function(d=-1, correlation_length=1)


def test_correlation_function_wrong_correlation_length_input():
    with pytest.raises(ValueError):
        correlation_function(d=1, correlation_length=-1)
    with pytest.raises(TypeError):
        correlation_function(d=1, correlation_length=[1, 2])


def test_inv_cov_vec_1D_scalar_and_vector_std_dev():
    """
    Check the diagonals of the inverse covariance matrix for scalar and vector
    standard deviation.
    """
    n_x = 10
    l_corr = 2.0
    coord_x = np.sort(np.linspace(0, 1, n_x) + np.random.rand(n_x) * 0.01)
    std_dev = np.random.rand(n_x) + np.finfo(float).eps

    # Correlation function
    def correlation_func(d):
        return correlation_function(
            d=d, correlation_length=l_corr, function_type="exponential"
        )

    # Standard deviation diagonal matrices for scalar and vector inputs
    std_dev_mx_from_scalar = np.diag(np.repeat(std_dev[0], n_x))
    std_dev_mx_from_vector = np.diag(std_dev)

    # Evaluate covariance matrices
    corr_mx = correlation_matrix(
        np.reshape(coord_x, (-1, 1)), correlation_func, distance_metric="euclidean"
    )
    cov_mx_scalar = np.matmul(
        np.matmul(std_dev_mx_from_scalar, corr_mx), std_dev_mx_from_scalar
    )
    cov_mx_vector = np.matmul(
        np.matmul(std_dev_mx_from_vector, corr_mx), std_dev_mx_from_vector
    )

    # Reference diagonals from direct inversion
    inv_cov_mx_scalar = np.linalg.inv(cov_mx_scalar)
    inv_cov_mx_vector = np.linalg.inv(cov_mx_vector)
    d0_scalar_ref = np.diag(inv_cov_mx_scalar)
    d1_scalar_ref = np.diag(inv_cov_mx_scalar, k=1)
    d0_vec_ref = np.diag(inv_cov_mx_vector)
    d1_vec_ref = np.diag(inv_cov_mx_vector, k=1)

    # Diagonals from inv_cov_vec_1D
    d0_scalar, d1_scalar = inv_cov_vec_1D(coord_x, l_corr, np.repeat(std_dev[0], n_x))
    d0_vec, d1_vec = inv_cov_vec_1D(coord_x, l_corr, std_dev)

    # Test
    np.testing.assert_allclose(d0_scalar, d0_scalar_ref)
    np.testing.assert_allclose(d1_scalar, d1_scalar_ref)
    np.testing.assert_allclose(d0_vec, d0_vec_ref)
    np.testing.assert_allclose(d1_vec, d1_vec_ref)


test_correlation_function_wrong_d_input()
test_correlation_function_wrong_correlation_length_input()
test_inv_cov_vec_1D_scalar_and_vector_std_dev()
print("Done")
