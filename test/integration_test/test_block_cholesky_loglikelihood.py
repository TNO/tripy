"""
Testing for the ND Cholesky loglikelihood with vector noise and associated functions:

    * symm_tri_block_chol: Block Cholesky decomposition of block symmetric
    tridiagonal matrices
"""

# ============================================================================
# Imports
# ============================================================================
import numpy as np
from scipy.stats import multivariate_normal

from tripy.kernels import Exponential
from tripy.loglikelihood import chol_loglike_2D, log_likelihood_linear_normal
from tripy.utils import inv_cov_vec_1D


def test_block_cholesky_loglikelihood():
    # =====================================================================
    # Build problem
    # =====================================================================

    # Parameters
    Nx = 50
    Nt = 5
    x = np.sort(np.linspace(0, 1, Nx) + np.random.rand(Nx) * 0.1)
    t = np.sort(np.linspace(0, 1, Nt) + np.random.rand(Nt) * 0.1)

    std_meas = np.repeat(1.0, Nx * Nt) + np.random.rand(Nx * Nt) + 0.1
    std_model = np.random.rand(Nt) + 0.1

    lcorr_x = 1.0
    lcorr_t = 1.0
    jitter = 1e-32

    # Initialize reference kernels
    kernel_x = Exponential(np.reshape(x, (-1, 1)), np.ones(Nx), length_scale=lcorr_x)
    kernel_t = Exponential(np.reshape(t, (-1, 1)), std_model, length_scale=lcorr_t)

    # Check covariance
    cov_mx_x = kernel_x.eval(np.ones(Nx)) + np.diag(np.ones(Nx) * jitter)
    cov_mx_t = kernel_t.eval(std_model)
    inv_cov_mx_x = np.linalg.inv(cov_mx_x)
    inv_cov_mx_t = np.linalg.inv(cov_mx_t)
    k_cov_mx = np.kron(cov_mx_t, cov_mx_x)
    e_cov_mx = np.diag(std_meas ** 2)

    # Model and measurements
    y_model = np.random.rand(Nx * Nt) + 5.0
    y_obs = np.random.rand(Nx * Nt) + 5.0

    # Get inverse cov vector in space and time
    Cx_0 = np.diag(inv_cov_mx_x)
    Cx_1 = np.diag(inv_cov_mx_x, k=1)
    Ct_0, Ct_1 = inv_cov_vec_1D(t, lcorr_t, std_model)

    # =====================================================================
    # Test block Cholesky likelihood function
    # =====================================================================

    def phys_model(theta):
        """
        Dummy physical model function for log_like_linear_normal
        """

        return theta

    # With scaling vector, tridiagonality in time and space
    # ---------------------------------------------------------------------
    Cx = [Cx_0, Cx_1]
    Ct = [Ct_0, Ct_1]

    # Mean and physical model eval
    y_phys_diag = np.diag(y_model)
    kph_cov_mx = np.matmul(np.matmul(y_phys_diag, k_cov_mx), y_phys_diag)
    cov = kph_cov_mx + e_cov_mx

    # Reference, linear normal and Cholesky solutions
    loglike_ref = multivariate_normal.logpdf(y_obs - y_model, cov=cov)
    loglike_linear_normal = log_likelihood_linear_normal(
        y_model, y_obs, phys_model, k_cov_mx, e_cov_mx
    )[0]
    loglike_chol = chol_loglike_2D(y_obs - y_model, y_model, Cx, Ct, std_meas)

    assert np.allclose(loglike_chol, loglike_linear_normal)
    assert np.allclose(loglike_chol, loglike_ref)

    # With scaling vector, tridiagonality in time only
    # ---------------------------------------------------------------------
    Cx = inv_cov_mx_x
    Ct = [Ct_0, Ct_1]

    # Mean and physical model eval
    y_phys_diag = np.diag(y_model)
    kph_cov_mx = np.matmul(np.matmul(y_phys_diag, k_cov_mx), y_phys_diag)
    cov = kph_cov_mx + e_cov_mx

    # Reference, linear normal and Cholesky solutions
    loglike_ref = multivariate_normal.logpdf(y_obs - y_model, cov=cov)
    loglike_linear_normal = log_likelihood_linear_normal(
        y_model, y_obs, phys_model, k_cov_mx, e_cov_mx
    )[0]
    loglike_chol = chol_loglike_2D(y_obs - y_model, y_model, Cx, Ct, std_meas)

    assert np.allclose(loglike_chol, loglike_linear_normal)
    assert np.allclose(loglike_chol, loglike_ref)

    # Inverted space and time
    # ---------------------------------------------------------------------

    # Model and observations with output shape of [Nx, Nt]
    y_model = np.transpose(np.random.rand(Nx, Nt) + 5.0)
    y_obs = np.transpose(np.random.rand(Nx, Nt) + 5.0)
    y_res = y_obs - y_model

    lcorr_x = 1.0
    lcorr_t = 1.0

    # Initialize kernels
    kernel_x = Exponential(np.reshape(x, (-1, 1)), np.ones(Nx), length_scale=lcorr_x)
    kernel_t = Exponential(np.reshape(t, (-1, 1)), std_model, length_scale=lcorr_t)

    # Calculate covariance matrices
    cov_mx_x = kernel_x.eval(np.ones(Nx)) + np.diag(np.ones(Nx) * jitter)
    cov_mx_t = kernel_t.eval(std_model)
    inv_cov_mx_x = np.linalg.inv(cov_mx_x)
    inv_cov_mx_t = np.linalg.inv(cov_mx_t) + np.diag(np.ones(Nt) * jitter)
    k_cov_mx = np.kron(cov_mx_x, cov_mx_t)
    e_cov_mx = np.diag(std_meas ** 2)

    Cx_0, Cx_1 = inv_cov_vec_1D(x, lcorr_x, np.ones(Nx))

    Cx = [Cx_0, Cx_1]
    Ct = inv_cov_mx_t

    # Mean and physical model eval
    y_phys_diag = np.diag(y_model.ravel())
    kph_cov_mx = np.matmul(np.matmul(y_phys_diag, k_cov_mx), y_phys_diag)
    cov = kph_cov_mx + e_cov_mx

    # Reference, linear normal and Cholesky solutions
    loglike_ref = multivariate_normal.logpdf(y_res.ravel(), cov=cov)
    loglike_linear_normal = log_likelihood_linear_normal(
        y_model.ravel(), y_obs.ravel(), phys_model, k_cov_mx, e_cov_mx
    )[0]
    loglike_chol = chol_loglike_2D(y_res.ravel(), y_model.ravel(), Ct, Cx, std_meas)

    assert np.allclose(loglike_chol, loglike_linear_normal)
    assert np.allclose(loglike_chol, loglike_ref)


test_block_cholesky_loglikelihood()
