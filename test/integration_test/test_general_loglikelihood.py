# -*- coding: utf-8 -*-
"""
Testing the proposed efficient loglikelihood evaluation in the 1D case.
"""

import numpy as np
from scipy import stats

from test.integration_test._test_utils import log_likelihood_reference
from tripy.kernels import Exponential
from tripy.loglikelihood import log_likelihood_linear_normal

jitter = 1e-16


def test_log_likelihood_linear_normal_multiplicative():
    # ============================================================================
    # Multiplicative model prediction error. The covariance matrix is composed
    # of a correlated model prediction error scaled by the model output, plus
    # an additive diagonal noise component.
    #
    # * Observations on unevenly spaced grid
    # * Noise std. dev. is a random vector
    # * Model std. dev. acts as a coefficient of variation, as the correlation
    # matrix is scaled by the model prediction.
    # ============================================================================
    N = 100
    lcorr = 10.0
    std_model = np.repeat(2.0, N)
    std_meas = np.random.rand(N) + 0.1

    # Coord vector and a test function
    x = np.sort(np.linspace(0, 1, N) + 0.1 * np.random.rand(N))

    # Example model
    def phys_model_func(x):
        return np.cos(x) + 5

    # Dummy model for reference loglikelihood function
    def dummy_model_func(theta):
        return theta

    # Model prediction, observations and residuals
    y_model = phys_model_func(x)
    y_obs = y_model + np.random.rand(N)
    y_res = y_obs - y_model

    # Assemble the covariance matrix for the reference solution
    e_cov_mx = np.diag(np.ones(N) * std_meas**2)
    kernel = Exponential(np.reshape(x, (-1, 1)), np.ones(N), length_scale=lcorr)
    k_cov_mx = kernel.eval(std_model, length_scale=lcorr)
    kph_cov_mx = np.matmul(np.diag(y_model), np.matmul(k_cov_mx, np.diag(y_model)))
    cov_mx = kph_cov_mx + e_cov_mx

    # Evaluate the loglikelihood
    logL_ref_1 = stats.multivariate_normal.logpdf(y_res, cov=cov_mx)
    logL_ref_2 = log_likelihood_reference(
        y_model.ravel(), y_obs.ravel(), dummy_model_func, k_cov_mx, e_cov_mx
    )[0]
    logL_test = log_likelihood_linear_normal(
        y_res, k_cov_mx, std_meas=std_meas, y_model=y_model
    )

    # Compare the two solutions
    print(logL_ref_1)
    print(logL_ref_2)
    print(logL_test)
    assert np.allclose(logL_ref_1, logL_test)
    assert np.allclose(logL_ref_2, logL_test)


def test_log_likelihood_linear_normal_multiplicative_no_noise():
    # ============================================================================
    # Multiplicative model prediction error. The covariance matrix is composed
    # of a correlated model prediction error scaled by the model output, plus
    # an additive diagonal noise component.
    #
    # * Observations on unevenly spaced grid
    # * Noise std. dev. is a random vector
    # * Model std. dev. acts as a coefficient of variation, as the correlation
    # matrix is scaled by the model prediction.
    # ============================================================================
    N = 100
    lcorr = 10.0
    std_model = np.repeat(2.0, N)

    # Coord vector and a test function
    x = np.sort(np.linspace(0, 1, N) + 0.1 * np.random.rand(N))

    # Example model
    def phys_model_func(x):
        return np.cos(x) + 5

    # Dummy model for reference loglikelihood function
    def dummy_model_func(theta):
        return theta

    # Model prediction, observations and residuals
    y_model = phys_model_func(x)
    y_obs = y_model + np.random.rand(N)
    y_res = y_obs - y_model

    # Assemble the covariance matrix for the reference solution
    kernel = Exponential(np.reshape(x, (-1, 1)), np.ones(N), length_scale=lcorr)
    k_cov_mx = kernel.eval(std_model, length_scale=lcorr)
    kph_cov_mx = np.matmul(np.diag(y_model), np.matmul(k_cov_mx, np.diag(y_model)))
    cov_mx = kph_cov_mx

    # Evaluate the loglikelihood
    logL_ref_1 = stats.multivariate_normal.logpdf(
        y_res, cov=cov_mx + np.diag(np.repeat(jitter, N))
    )
    logL_ref_2 = log_likelihood_reference(
        y_model.ravel(), y_obs.ravel(), dummy_model_func, k_cov_mx
    )[0]
    logL_test = log_likelihood_linear_normal(y_res, k_cov_mx, y_model=y_model)

    # Compare the two solutions
    print(logL_ref_1)
    print(logL_ref_2)
    print(logL_test)
    assert np.allclose(logL_ref_1, logL_test)
    assert np.allclose(logL_ref_2, logL_test)


def test_log_likelihood_linear_normal_additive():
    # ============================================================================
    #     # Additive model prediction error. The covariance is composed of a
    #     # correlated model prediction uncertainty defined by the std. dev.
    #     # plus an additive diagonal noise component.
    #     #
    #     # * Observations on unevenly spaced grid
    #     # * Noise std. dev. is a random vector
    #     # * Model std. dev. is the standard deviation of the additive model
    #     # prediction uncerainty.
    # ============================================================================
    N = 100
    lcorr = 10.0
    std_model = np.repeat(2.0, N)
    std_meas = np.random.rand(N) + 0.1

    # Coord vector and a test function
    x = np.sort(np.linspace(0, 1, N) + 0.1 * np.random.rand(N))

    # Example model
    def phys_model_func(x):
        return np.cos(x) + 5

    # Dummy model for reference loglikelihood function
    def dummy_model_func(theta):
        return theta

    # Model prediction, observations and residuals
    y_model = phys_model_func(x)
    y_obs = y_model + np.random.rand(N)
    y_res = y_obs - y_model

    # Assemble the covariance matrix for the reference solution
    e_cov_mx = np.diag(np.ones(N) * std_meas**2)
    kernel = Exponential(np.reshape(x, (-1, 1)), np.ones(N), length_scale=lcorr)
    k_cov_mx = kernel.eval(std_model, length_scale=lcorr)
    cov_mx = k_cov_mx + e_cov_mx

    # Evaluate the loglikelihood
    logL_ref_1 = stats.multivariate_normal.logpdf(y_res, cov=cov_mx)
    logL_ref_2 = log_likelihood_reference(
        y_model.ravel(), y_obs.ravel(), dummy_model_func, e_cov_mx=k_cov_mx + e_cov_mx
    )[0]
    logL_test = log_likelihood_linear_normal(y_res, k_cov_mx, std_meas=std_meas)

    # Compare the two solutions
    print(logL_ref_1)
    print(logL_ref_2)
    print(logL_test)
    assert np.allclose(logL_ref_1, logL_test)
    assert np.allclose(logL_ref_2, logL_test)


def test_log_likelihood_linear_normal_additive_no_noise():
    # ============================================================================
    #     # Additive model prediction error. The covariance is composed of a
    #     # correlated model prediction uncertainty defined by the std. dev.
    #     # plus an additive diagonal noise component.
    #     #
    #     # * Observations on unevenly spaced grid
    #     # * Noise std. dev. is a random vector
    #     # * Model std. dev. is the standard deviation of the additive model
    #     # prediction uncerainty.
    # ============================================================================
    N = 100
    lcorr = 10.0
    std_model = np.repeat(2.0, N)

    # Coord vector and a test function
    x = np.sort(np.linspace(0, 1, N) + 0.1 * np.random.rand(N))

    # Example model
    def phys_model_func(x):
        return np.cos(x) + 5

    # Dummy model for reference loglikelihood function
    def dummy_model_func(theta):
        return theta

    # Model prediction, observations and residuals
    y_model = phys_model_func(x)
    y_obs = y_model + np.random.rand(N)
    y_res = y_obs - y_model

    # Assemble the covariance matrix for the reference solution
    kernel = Exponential(np.reshape(x, (-1, 1)), np.ones(N), length_scale=lcorr)
    k_cov_mx = kernel.eval(std_model, length_scale=lcorr)
    cov_mx = k_cov_mx

    # Evaluate the loglikelihood
    logL_ref_1 = stats.multivariate_normal.logpdf(y_res, cov=cov_mx)
    logL_ref_2 = log_likelihood_reference(
        y_model.ravel(), y_obs.ravel(), dummy_model_func, e_cov_mx=k_cov_mx
    )[0]
    logL_test = log_likelihood_linear_normal(y_res, k_cov_mx)

    # Compare the two solutions
    print(logL_ref_1)
    print(logL_ref_2)
    print(logL_test)
    assert np.allclose(logL_ref_1, logL_test)
    assert np.allclose(logL_ref_2, logL_test)


# def test_chol_loglike_1D_additive():
#     # ============================================================================
#     # Additive model prediction error. The covariance is composed of a
#     # correlated model prediction uncertainty defined by the std. dev.
#     # plus an additive diagonal noise component.
#     #
#     # * Observations on unevenly spaced grid
#     # * Noise std. dev. is a random vector
#     # * Model std. dev. is the standard deviation of the additive model
#     # prediction uncerainty.
#     # ============================================================================
#     N = 100
#     lcorr = 10.0
#     std_model = np.repeat(2.0, N)
#     std_meas = np.random.rand(N) + 0.1
#
#     # Coord vector and a test function
#     x = np.sort(np.linspace(0, 1, N) + 0.1 * np.random.rand(N))
#
#     # In the case of additive model prediction uncertainty, a vector of ones
#     # is given as input to chol_loglike_1D instead of model predictions
#     y_model = np.ones(N)
#
#     # Vector of residuals between measurement and prediction
#     y_res = np.random.rand(N)
#
#     # ============================================================================
#     # Compare 1D Cholesky with reference solution
#     # ============================================================================
#
#     # Assemble the covariance matrix for the reference solution
#     e_cov_mx = np.diag(np.ones(N) * std_meas ** 2)
#     kernel = Exponential(np.reshape(x, (-1, 1)), np.ones(N), length_scale=lcorr)
#     corr_mx = kernel.eval(std_model, length_scale=lcorr)
#     cov_mx = corr_mx + e_cov_mx
#
#     # Evaluate the loglikelihood
#     logL_ref = stats.multivariate_normal.logpdf(y_res, cov=cov_mx)
#     logL_chol = chol_loglike_1D(
#         y_res, x, lcorr, std_model, std_meas=std_meas, y_model=y_model
#     )
#
#     # Compare the two solutions
#     print(logL_ref)
#     print(logL_chol)
#     assert np.allclose(logL_ref, logL_chol)
#
#
# def test_chol_loglike_1D_no_noise():
#     # ============================================================================
#     # Multiplicative, no measurement uncertainty
#     #
#     # * Observations on unevenly spaced grid
#     # * Model std. dev. is the standard deviation of the additive model
#     # prediction uncerainty.
#     # ============================================================================
#     N = 100
#     lcorr = 10.0
#     std_model = np.repeat(2.0, N)
#
#     # Coord vector and a test function
#     x = np.sort(np.linspace(0, 1, N) + 0.1 * np.random.rand(N))
#
#     # Example model
#     def phys_model_func(x):
#         return np.cos(x) + 5
#
#     # Model prediction
#     y_model = phys_model_func(x)
#
#     # Vector of residuals between measurement and prediction
#     y_res = np.random.rand(N)
#
#     # ============================================================================
#     # Compare 1D Cholesky with reference solution
#     # ============================================================================
#
#     # Assemble the covariance matrix for the reference solution
#     kernel = Exponential(np.reshape(x, (-1, 1)), np.ones(N), length_scale=lcorr)
#     corr_mx = kernel.eval(std_model, length_scale=lcorr)
#     kph_cov_mx = np.matmul(np.diag(y_model), np.matmul(corr_mx, np.diag(y_model)))
#
#     # Evaluate the loglikelihood
#     logL_ref = stats.multivariate_normal.logpdf(y_res, cov=kph_cov_mx)
#     logL_chol = chol_loglike_1D(y_res, x, lcorr, std_model, y_model=y_model)
#
#     # Compare the two solutions
#     print(logL_ref)
#     print(logL_chol)
#     assert np.allclose(logL_ref, logL_chol)
#
#     # ============================================================================
#     # Additive, no measurement uncertainty
#     #
#     # * Observations on unevenly spaced grid
#     # * Model std. dev. is the standard deviation of the additive model
#     # prediction uncerainty.
#     # ============================================================================
#     N = 100
#     lcorr = 10.0
#     std_model = np.repeat(2.0, N)
#
#     # Coord vector and a test function
#     x = np.sort(np.linspace(0, 1, N) + 0.1 * np.random.rand(N))
#
#     # Vector of residuals between measurement and prediction
#     y_res = np.random.rand(N)
#
#     # ============================================================================
#     # Compare 1D Cholesky with reference solution
#     # ============================================================================
#
#     # Assemble the covariance matrix for the reference solution
#     kernel = Exponential(np.reshape(x, (-1, 1)), np.ones(N), length_scale=lcorr)
#     corr_mx = kernel.eval(std_model, length_scale=lcorr)
#
#     # Evaluate the loglikelihood
#     logL_ref = stats.multivariate_normal.logpdf(y_res, cov=corr_mx)
#     logL_chol = chol_loglike_1D(y_res, x, lcorr, std_model)
#
#     # Compare the two solutions
#     print(logL_ref)
#     print(logL_chol)
#     assert np.allclose(logL_ref, logL_chol)
