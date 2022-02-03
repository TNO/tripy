"""
Testing for the Kronecker loglikelihood functions
"""

# ============================================================================
# Imports
# ============================================================================
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from tripy.kernels import Exponential, RBF
from tripy.loglikelihood import _kron_loglike_ND_tridiag, kron_loglike_2D
from tripy.utils import inv_cov_vec_1D

jitter = 1e-6


def test_kron_loglike_2D_tridiagonal():
    """
    Test the `kron_loglike_2D` function in the case where both
    dimensions have a tridiagonal inverse correlation matrix.
    """
    # Grid parameters
    Nx = 3
    Nt = 20
    x = np.sort(np.linspace(0, 1, Nx) + np.random.rand(Nx) * 0.1)
    t = np.sort(np.linspace(0, 1, Nt) + np.random.rand(Nt) * 0.1)

    # Noise parameters
    std_meas = 1.0  # np.random.rand() + 1.0
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
    Nt = 3
    x = np.sort(np.linspace(0, 1, Nx) + np.random.rand(Nx) * 0.1)
    t = np.sort(np.linspace(0, 1, Nt) + np.random.rand(Nt) * 0.1)

    # Noise parameters
    std_meas = 0.001  # np.random.rand() + 1.0
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
    Nx = 20
    Nt = 3
    x = np.sort(np.linspace(0, 1, Nx) + np.random.rand(Nx) * 0.1)
    t = np.sort(np.linspace(0, 1, Nt) + np.random.rand(Nt) * 0.1)

    # Noise parameters
    std_meas = 0.001  # np.random.rand() + 1.0
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
    std_meas = 0.001  # np.random.rand() + 1.0
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


@pytest.mark.skip
def test_kron_loglike_ND_tridiagonal():
    """
    Test the `kron_loglike_ND` function in the case where no
    dimension is tridiagonal.
    """

    # Initialize
    Nx = [5, 5, 5, 5]
    ND = len(Nx)
    N = np.prod(Nx)

    # Assemble inputs
    lcorr = np.random.rand(ND) + 1.0
    std_meas = np.random.rand() + 100.0

    x = [np.sort(np.linspace(0, 1, Nxi) + np.random.rand(Nxi) * 0.1) for Nxi in Nx]
    std_model = [np.repeat(np.random.rand() + 1.0, Nxi) for Nxi in Nx]
    kernels = [
        Exponential(np.reshape(xi, (-1, 1)), std_model[i], length_scale=lcorr[i])
        for i, xi in enumerate(x)
    ]

    # Evaluate and invert covariance matrices for each dimension
    k_cov_mx_list = [
        kernel.eval(std_model[i], length_scale=lcorr[i])
        for i, kernel in enumerate(kernels)
    ]
    # k_cov_mx_inv = [np.linalg.inv(k_cov_i) for k_cov_i in k_cov_mx_list]

    # Assemble full covariance for reference solution
    k_cov_mx = np.kron(k_cov_mx_list[1], k_cov_mx_list[0])
    for i in range(2, ND):
        k_cov_mx = np.kron(k_cov_mx_list[i], k_cov_mx)
    cov_mx = k_cov_mx + np.diag(np.repeat(std_meas ** 2, N))
    cov_mx_jit = k_cov_mx + np.diag(np.repeat(jitter ** 2, N))

    # Array of observations
    y = 10 * np.random.rand(N)

    # Evaluate loglikelihood
    loglike_ref_noiseless = multivariate_normal.logpdf(np.ravel(y), cov=cov_mx_jit)
    loglike_test_noiseless = _kron_loglike_ND_tridiag(y, x, std_model, lcorr)
    loglike_ref_noisy = multivariate_normal.logpdf(np.ravel(y), cov=cov_mx)
    loglike_test_noisy = _kron_loglike_ND_tridiag(
        y, x, std_model, lcorr, std_meas=std_meas
    )

    print(loglike_ref_noiseless)
    print(loglike_test_noiseless)
    print(loglike_ref_noisy)
    print(loglike_test_noisy)
    assert np.allclose(loglike_test_noiseless, loglike_ref_noiseless)
    assert np.allclose(loglike_test_noisy, loglike_ref_noisy)


def test_kron_loglike_ND_general():
    """
    Test the `kron_loglike_ND` function in the case where all
    dimensions are tridiagonal
    """


test_kron_loglike_2D_tridiagonal()
test_kron_loglike_2D_mixed_1()
test_kron_loglike_2D_mixed_2()
test_kron_loglike_2D_general()
test_kron_loglike_ND_tridiagonal()
