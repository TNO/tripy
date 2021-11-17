# -*- coding: utf-8 -*-
"""
@author: kounei

Testing the proposed efficient loglikelihood evaluation in the 1D case.
"""

import numpy as np
from scipy import stats

from tripy.kernels import Exponential
from tripy.loglikelihood import chol_loglike_1D


def test_chol_loglike_1D():
    # ============================================================================
    # SETUP
    #
    # * Observations on unevenly spaced grid
    # * Noise std. dev. is a random vector
    # ============================================================================
    N = 100
    lcorr = 10.0
    std_model = np.repeat(2.0, N)
    std_meas = np.random.rand(N) + 0.1  # np.repeat(2.0, N)

    # Coord vector and a test function
    x = np.sort(np.linspace(0, 1, N) + 0.1 * np.random.rand(N))

    # Example model
    def phys_model_func(x):
        return np.cos(x) + 5

    # Synthetic observations
    y_model = phys_model_func(x)
    v_obs = np.random.rand(N) + y_model

    # ============================================================================
    # Compare 1D Cholesky with reference solution
    # ============================================================================

    # Assemble the covariance matrix
    e_cov_mx = np.diag(np.ones(N) * std_meas ** 2)
    kernel = Exponential(np.reshape(x, (-1, 1)), np.ones(N), length_scale=lcorr)
    corr_mx = kernel.eval(std_model, length_scale=lcorr)
    kph_cov_mx = np.matmul(np.diag(y_model), np.matmul(corr_mx, np.diag(y_model)))
    cov_mx = kph_cov_mx + e_cov_mx

    # Evaluate the loglikelihood
    logL_ref = stats.multivariate_normal.logpdf(v_obs, cov=cov_mx)
    logL_chol = chol_loglike_1D(x, y_model, v_obs, lcorr, std_model, std_meas)

    # Compare the two solutions
    print(logL_ref)
    print(logL_chol)
    assert np.allclose(logL_ref, logL_chol)
