"""
Testing for the Kronecker loglikelihood functions
"""

# ============================================================================
# Imports
# ============================================================================
import numpy as np
from scipy.stats import multivariate_normal

from tripy.kernels import Exponential, RBF
from tripy.loglikelihood import (
    kron_loglike_2D,
)
from tripy.utils import inv_cov_vec_1D

jitter = 1e-6


def test_kron_loglike_2D_tridiagonal():
    """
    Test the `kron_loglike_2D` function in the case where both
    dimensions have a tridiagonal inverse correlation matrix.
    """
    # Grid parameters
    Nx = 20
    Nt = 20
    x = np.sort(np.linspace(0, 1, Nx) + np.random.rand(Nx) * 0.1)
    t = np.sort(np.linspace(0, 1, Nt) + np.random.rand(Nt) * 0.1)

    # Noise parameters
    std_meas = np.random.rand() + 1.0
    std_t = np.repeat(2.0, Nt)

    # Correlation parameters
    lcorr_x = 1.0
    lcorr_t = 1.0

    # Initialize reference kernels
    kernel_x = Exponential(np.reshape(x, (-1, 1)), np.ones(Nx), length_scale=lcorr_x)
    kernel_t = Exponential(
        np.reshape(t, (-1, 1)), np.repeat(std_t, Nt), length_scale=lcorr_t
    )

    # Assemble covariance matrices for reference loglikelihood
    cov_mx_x = kernel_x.eval(np.ones(Nx))
    cov_mx_t = kernel_t.eval(std_t)
    k_cov_mx = np.kron(cov_mx_x, cov_mx_t)
    e_cov_mx = np.diag(np.repeat(std_meas ** 2, Nx * Nt))

    Ct_0, Ct_1 = inv_cov_vec_1D(t, lcorr_t, std_t)
    Cx_0, Cx_1 = inv_cov_vec_1D(x, lcorr_x, np.ones(Nx))

    # Model and measurements
    y_model = np.random.rand(Nx, Nt) + 5.0
    y_obs = np.random.rand(Nx, Nt) + 5.0
    y_res = y_obs - y_model

    # Reference covariance matrix
    cov = k_cov_mx + e_cov_mx
    cov_jit = k_cov_mx + np.diag(np.repeat(jitter, Nx * Nt))

    # Reference, linear normal and Cholesky solutions
    loglike_ref_noiseless = multivariate_normal.logpdf(y_res.ravel(), cov=cov_jit)
    loglike_test_noiseless = kron_loglike_2D(y_res, [Cx_0, Cx_1], [Ct_0, Ct_1])
    loglike_ref_noisy = multivariate_normal.logpdf(y_res.ravel(), cov=cov)
    loglike_test_noisy = kron_loglike_2D(
        y_res, [Cx_0, Cx_1], [Ct_0, Ct_1], std_meas=std_meas
    )

    print(loglike_ref_noiseless)
    print(loglike_test_noiseless)
    print(loglike_ref_noisy)
    print(loglike_test_noisy)
    assert np.allclose(loglike_test_noiseless, loglike_ref_noiseless)
    assert np.allclose(loglike_test_noisy, loglike_ref_noisy)


def test_kron_loglike_2D_mixed_1():
    """
    Test the `kron_loglike_2D` function in the case where only
    dimension 1 is tridiagonal.
    """
    # Grid parameters
    Nx = 20
    Nt = 20
    x = np.sort(np.linspace(0, 1, Nx) + np.random.rand(Nx) * 0.1)
    t = np.sort(np.linspace(0, 1, Nt) + np.random.rand(Nt) * 0.1)

    # Noise parameters
    std_meas = np.random.rand() + 1.0
    std_t = np.repeat(2.0, Nt)

    # Correlation parameters
    lcorr_x = 1.0
    lcorr_t = 1.0

    # Initialize reference kernels
    kernel_x = RBF(np.reshape(x, (-1, 1)), np.ones(Nx), length_scale=lcorr_x)
    kernel_t = Exponential(
        np.reshape(t, (-1, 1)), np.repeat(std_t, Nt), length_scale=lcorr_t
    )

    # Assemble covariance matrices for reference loglikelihood
    cov_mx_x = kernel_x.eval(np.ones(Nx))
    cov_mx_t = kernel_t.eval(std_t)
    k_cov_mx = np.kron(cov_mx_x, cov_mx_t)
    e_cov_mx = np.diag(np.repeat(std_meas ** 2, Nx * Nt))

    Ct_0, Ct_1 = inv_cov_vec_1D(t, lcorr_t, std_t)

    # Model and measurements
    y_model = np.random.rand(Nx, Nt) + 5.0
    y_obs = np.random.rand(Nx, Nt) + 5.0
    y_res = y_obs - y_model

    # Reference covariance matrix
    cov = k_cov_mx + e_cov_mx
    cov_jit = k_cov_mx + np.diag(np.repeat(jitter, Nx * Nt))

    # Reference, linear normal and Cholesky solutions
    loglike_ref_noiseless = multivariate_normal.logpdf(y_res.ravel(), cov=cov_jit)
    loglike_test_noiseless = kron_loglike_2D(y_res, cov_mx_x, [Ct_0, Ct_1])
    loglike_ref_noisy = multivariate_normal.logpdf(y_res.ravel(), cov=cov)
    loglike_test_noisy = kron_loglike_2D(
        y_res, cov_mx_x, [Ct_0, Ct_1], std_meas=std_meas
    )

    print(loglike_ref_noiseless)
    print(loglike_test_noiseless)
    print(loglike_ref_noisy)
    print(loglike_test_noisy)
    assert np.allclose(loglike_test_noiseless, loglike_ref_noiseless)
    assert np.allclose(loglike_test_noisy, loglike_ref_noisy)


def test_kron_loglike_2D_mixed_2():
    """
    Test the `kron_loglike_2D` function in the case where only
    dimension 2 is tridiagonal.
    """
    # Grid parameters
    Nx = 10
    Nt = 10
    x = np.sort(np.linspace(0, 1, Nx) + np.random.rand(Nx) * 0.1)
    t = np.sort(np.linspace(0, 1, Nt) + np.random.rand(Nt) * 0.1)

    # Noise parameters
    std_meas = np.random.rand() + 1.0
    std_x = np.repeat(2.0, Nx)

    # Correlation parameters
    lcorr_x = 1.0
    lcorr_t = 1.0

    # Initialize reference kernels
    kernel_x = Exponential(
        np.reshape(x, (-1, 1)), np.repeat(std_x, Nx), length_scale=lcorr_x
    )
    kernel_t = RBF(np.reshape(t, (-1, 1)), np.ones(Nt), length_scale=lcorr_t)

    # Assemble covariance matrices for reference loglikelihood
    cov_mx_x = kernel_x.eval(std_x)
    cov_mx_t = kernel_t.eval(np.ones(Nt))
    k_cov_mx = np.kron(cov_mx_x, cov_mx_t)
    e_cov_mx = np.diag(np.repeat(std_meas ** 2, Nx * Nt))

    Cx_0, Cx_1 = inv_cov_vec_1D(x, lcorr_x, std_x)

    # Model and measurements
    y_model = np.random.rand(Nx, Nt) + 5.0
    y_obs = np.random.rand(Nx, Nt) + 5.0
    y_res = y_obs - y_model

    # Reference covariance matrix
    cov = k_cov_mx + e_cov_mx
    cov_jit = k_cov_mx + np.diag(np.repeat(jitter, Nx * Nt))

    # Reference, linear normal and Cholesky solutions
    loglike_ref_noiseless = multivariate_normal.logpdf(y_res.ravel(), cov=cov_jit)
    loglike_test_noiseless = kron_loglike_2D(y_res, [Cx_0, Cx_1], cov_mx_t)
    loglike_ref_noisy = multivariate_normal.logpdf(y_res.ravel(), cov=cov)
    loglike_test_noisy = kron_loglike_2D(y_res, [Cx_0, Cx_1], cov_mx_t, std_meas)

    print(loglike_ref_noiseless)
    print(loglike_test_noiseless)
    print(loglike_ref_noisy)
    print(loglike_test_noisy)
    assert np.allclose(loglike_test_noiseless, loglike_ref_noiseless)
    assert np.allclose(loglike_test_noisy, loglike_ref_noisy)


def test_kron_loglike_2D_general():
    """
    Test the `kron_loglike_2D` function in the case where no
    dimension is tridiagonal.
    """

    # Grid parameters
    Nx = 20
    Nt = 20
    x = np.sort(np.linspace(0, 1, Nx) + np.random.rand(Nx) * 0.1)
    t = np.sort(np.linspace(0, 1, Nt) + np.random.rand(Nt) * 0.1)

    # Noise parameters
    std_meas = np.random.rand() + 1.0
    std_t = np.repeat(2.0, Nt)

    # Correlation parameters
    lcorr_x = 1.0
    lcorr_t = 1.0

    # Initialize reference kernels
    kernel_x = RBF(np.reshape(x, (-1, 1)), np.ones(Nx), length_scale=lcorr_x)
    kernel_t = RBF(np.reshape(t, (-1, 1)), np.repeat(std_t, Nt), length_scale=lcorr_t)

    # Assemble covariance matrices for reference loglikelihood
    cov_mx_x = kernel_x.eval(np.ones(Nx))
    cov_mx_t = kernel_t.eval(std_t)
    k_cov_mx = np.kron(cov_mx_x, cov_mx_t)
    e_cov_mx = np.diag(np.repeat(std_meas ** 2, Nx * Nt))

    # Model and measurements
    y_model = np.random.rand(Nx, Nt) + 5.0
    y_obs = np.random.rand(Nx, Nt) + 5.0
    y_res = y_obs - y_model

    # Reference covariance matrix
    cov = k_cov_mx + e_cov_mx
    cov_jit = k_cov_mx + np.diag(np.repeat(jitter, Nx * Nt))

    # Reference, linear normal and Cholesky solutions
    loglike_ref_noiseless = multivariate_normal.logpdf(y_res.ravel(), cov=cov_jit)
    loglike_test_noiseless = kron_loglike_2D(y_res, cov_mx_x, cov_mx_t)
    loglike_ref_noisy = multivariate_normal.logpdf(y_res.ravel(), cov=cov)
    loglike_test_noisy = kron_loglike_2D(y_res, cov_mx_x, cov_mx_t, std_meas=std_meas)

    print(loglike_ref_noiseless)
    print(loglike_test_noiseless)
    print(loglike_ref_noisy)
    print(loglike_test_noisy)
    assert np.allclose(loglike_test_noiseless, loglike_ref_noiseless)
    assert np.allclose(loglike_test_noisy, loglike_ref_noisy)


# test_kron_loglike_2D_tridiagonal()
test_kron_loglike_2D_mixed_1()
test_kron_loglike_2D_mixed_2()
test_kron_loglike_2D_general()
