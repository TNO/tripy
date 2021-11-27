"""
Likelihood evaluation
"""

from typing import Callable, List, Optional, Union

import numpy as np
import torch
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.linalg.lapack import dpotrf
from torch.distributions import MultivariateNormal

from tripy.utils import (
    cho_solve_symm_tridiag,
    chol_tridiag,
    inv_cov_vec_1D,
    kron_op,
    symm_tri_block_chol,
    symm_tri_block_solve,
)


def chol_loglike_2D(y_res, y_model, Cx, Ct, std_meas):
    """
    Given the uncertainty parameters and vectors of the x and t positions
    on a lattice, calculates the loglikelihood of the 2D Gaussian process
    with multiplicative space and time uncertainties and i.i.d. noise.

    TODO :
        * Find a way to replace DPOTRF with DPTTRF in the Cholesky decomposition
            to take advantage of tridiagonality
        * Add some basic input checks
        * It is intended that this function is general and works with any combination
            of spatial and temporal covariance, with only the following restrictions:
                - Space and time are separable (Kronecker structure)
                - Correlation in time is exponential
            If possible, this function should be unaware of the types of kernels passed
            and call methods of the kernel class (e.g. kernel.inv, kernel.chol...)
            and the rest will be handled by the kernel. This would allow for taking
            advantage of kernel-specific optimizations without overcomplicating this
            function.
        * Implement the following methods for the kernels (when more efficient than
        the general case):
            - inv
            - inv_chol (with option of adding a diagonal vector)
            - inv_eig

    INPUT:
        y_res: [Nx, Nt] Array of residuals
        y_model: [Nx, Nt] Array of model output
        x, t: [1, Nx], [1, Nt] Space and time point vectors
        lcorr: Space and time correlation lengthscale
        std: Standard deviation
        chol: Callable

    OPTIONAL:
        check_finite: Optional flag of scipy.linalg.eigh_tridiagonal. False improves
        performance

    RETURNS:
        L: Loglikelihood assuming exponential covariance and multiplicative
        modeling uncertainty
    """

    # Check if Cx and Ct are lists or numpy arrays.
    if isinstance(Cx, list):
        # TODO: Change dpttrf to chol_tridiag
        Nx = len(Cx[0])
        Lx = chol_tridiag(Cx[0], Cx[1])[0]
    else:
        Nx = np.shape(Cx)[0]
        Lx = np.diag(dpotrf(Cx, lower=1, clean=1, overwrite_a=0)[0])

    if isinstance(Ct, list):
        Nt = len(Ct[0])
        Lt = chol_tridiag(Ct[0], Ct[1])[0]
    else:
        raise ValueError(
            "Ct must be a list containing the diagonal and "
            "off-diagonal vectors of a tridiagonal matrix"
        )

    L, C = symm_tri_block_chol(Cx, Ct, std_meas ** 2, y=y_model)

    # Get diagonal elements of L
    # TODO: This will have to be modified to extract the diagonal in both the case that
    #   the inverse of cov_mx_x is tridiagonal and in the general case.
    Ldiag = np.diagonal(L, axis1=1, axis2=2)

    # Vectors to be used later
    Winv_vec = np.ones(Nx * Nt) / std_meas ** 2
    yWy = np.sum(y_res ** 2 * (1 / std_meas ** 2))
    WGx = Winv_vec * y_model.ravel() * y_res.ravel()

    # Solve the linear system
    X = symm_tri_block_solve(L, C, WGx, Nx, Nt)

    # Vector product
    xSx = yWy - np.sum(WGx * X)

    # ========================================================================
    # Logdeterminant
    # ========================================================================

    # Logdet of noise matrix
    logdet_W = np.sum(2 * np.log(std_meas))

    # Logdet of the full correlation matrix
    logdet_C = -2 * np.sum(np.log(Lx)) * Nt + -2 * np.sum(np.log(Lt)) * Nx

    # Logdet of Cholesky factors
    logdet_chol = 2 * np.sum(np.log(Ldiag))
    logdet_tot = logdet_W + logdet_C + logdet_chol

    return -0.5 * (logdet_tot + xSx + Nx * Nt * np.log(2 * np.pi))


def log_likelihood_linear_normal(
    theta: np.ndarray,
    x_meas: np.ndarray,
    physical_model: Callable[[np.ndarray], np.ndarray],
    k_cov_mx: Optional[np.ndarray] = None,
    e_cov_mx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Log-likelihood function for the following model:

    ``X_model = K*physical_model_fun(theta) + E``

    where
        * ``K`` ~ MVN(mean=1, covariance=k_cov_mx)
        * ``E`` ~ MVN(mean=0, covariance=e_cov_mx)
        * ``MVN()`` multivariate normal distribution

    ``physical_model_fun()`` has no additional uncertainty, e.g. surrogate model
    uncertainty.

    Args:
        theta: value of the parameter(s) to be estimated in the Bayesian inference,
            `shape [1, K].
        x_meas: measurements in a [T, S] shape. T corresponds to the time space and S
            to the physical and quantity space.
        physical_model: function that takes as input theta and gives out x_model.
        k_cov_mx: covariance matrix of `K`, shape [T*S, T*S].
        e_cov_mx: covariance matrix of `elastic_mod`, shape [T*S, T*S].

    Returns:
        log_like: value of the log-likelihood evaluated at theta.

    """
    # ..................................................................................
    # Pre-process
    # ..................................................................................
    x_meas = x_meas.ravel()
    theta = np.atleast_2d(theta)
    n_theta = theta.shape[0]

    x_meas_n_elements = x_meas.size
    k_cov_mx_zero_flag = False

    if k_cov_mx is None:
        k_cov_mx = np.zeros((x_meas_n_elements, x_meas_n_elements))
        kph_cov_mx = k_cov_mx
        k_cov_mx_zero_flag = True

    if e_cov_mx is None:
        e_cov_mx = np.zeros((x_meas_n_elements, x_meas_n_elements))

    # ..................................................................................
    # Calculate likelihood value(s)
    # ..................................................................................
    log_like = np.empty(n_theta)
    for ii, theta_row in enumerate(theta):
        x_ph = physical_model(theta_row).ravel()
        x_ph_vector = x_ph.reshape(-1)
        x_ph_diag_mx = np.diag(x_ph_vector)

        x_diff = torch.tensor(x_meas - x_ph)

        # covariance matrix of K*physical_model(theta)
        if not k_cov_mx_zero_flag:
            kph_cov_mx = np.matmul(x_ph_diag_mx, np.matmul(k_cov_mx, x_ph_diag_mx))

        mvn = MultivariateNormal(
            torch.zeros(len(x_ph_vector)), torch.tensor(kph_cov_mx + e_cov_mx)
        )
        log_like[ii] = mvn.log_prob(x_diff).detach().cpu().numpy()
    return log_like


def kron_loglike_2D_tridiag(
    y, x, t, std_meas, l_corr_x, std_x, l_corr_t, std_t, check_finite=False
):
    """
    Given the uncertainty parameters and vectors of the x and t positions
    on a lattice, calculates the loglikelihood of the observations
    for exponential space and time correlation in the additive modeling
    uncertainty and i.i.d. Gaussian noise

    WARNING: This is superceded by kron_loglike_ND_tridiag and will be
    removed.

    INPUT:
        y: [Nx, Nt] Array of observations
        x, t: [1, Nx], [1, Nt] Space and time point vectors
        lcorr: Space and time correlation lengthscale
        std: Standard deviation

    OPTIONAL:
        check_finite: Optional flag of scipy.linalg.eigh_tridiagonal. False improves
        performance

    RETURNS:
        L: Loglikelihood assuming exponential covariance and additive modeling
        uncertainty
    """

    Nx = len(x)
    Nt = len(t)

    Cx_0, Cx_1 = inv_cov_vec_1D(x, l_corr_x, std_x)
    Ct_0, Ct_1 = inv_cov_vec_1D(t, l_corr_t, std_t)

    # Eigendecomposition using tridiagonality
    lambda_t, w_t = eigh_tridiagonal(Ct_0, Ct_1, check_finite=check_finite)
    lambda_x, w_x = eigh_tridiagonal(Cx_0, Cx_1, check_finite=check_finite)

    # Kronecker prod of eigenvalues. This is the main diagonal of the diagonal
    # covariance matrix
    C_xt = np.kron(1 / lambda_x, 1 / lambda_t) + std_meas ** 2

    # Determinant: C_xt is diagonal. We can therefore sum the log terms. This is the
    # determinant of the inverse.
    logdet_C_xt = np.sum(np.log(C_xt))

    # Rotated data vector.
    Y = np.ravel(np.matmul(np.matmul(w_x.T, y), w_t))

    # Loglikelihood
    return (
        -Nx * Nt / 2 * np.log(2 * np.pi)
        - 0.5 * logdet_C_xt
        - 0.5 * np.sum(Y ** 2 * 1 / C_xt)
    )


def kron_loglike_temporal(
    y, std_meas, cov_mx_x, t, l_corr_t, std_t, check_finite=False
):
    """
    Efficient loglikelihood evaluation for Kronecker covariance matrices
    with Markovian temporal covariance.

    Any combination of a single dimension with Markovian covariance and N
    nonseparable dimensions with non-Markovian covariance can be evaluated
    using this function.

    INPUT:
        y: [Nx, Nt] Array of observations
        t: [1, Nt] Time point vector
        lcorr: Time correlation lengthscale
        std: Standard deviation

    OPTIONAL:
        check_finite: Optional flag of scipy.linalg.eigh_tridiagonal. False improves
        performance

    RETURNS:
        L: Loglikelihood assuming exponential covariance and additive modeling
        uncertainty
    """

    Nx = np.shape(cov_mx_x)[0]
    Nt = len(t)
    Ct_0, Ct_1 = inv_cov_vec_1D(t, l_corr_t, std_t)

    # Eigendecomposition using tridiagonality
    lambda_t, w_t = eigh_tridiagonal(Ct_0, Ct_1, check_finite=check_finite)
    lambda_x, w_x = eigh(cov_mx_x, check_finite=check_finite)

    # Kronecker prod of eigenvalues.
    C_xt = np.kron(lambda_x, 1 / lambda_t) + std_meas ** 2

    # Determinant: C_xt is diagonal. We can therefore sum the log terms. This is the
    # determinant of the inverse.
    logdet_C_xt = np.sum(np.log(C_xt))

    # Rotated data vector.
    Y = np.ravel(np.matmul(np.matmul(w_x.T, y), w_t))

    # Loglikelihood
    return (
        -Nx * Nt / 2 * np.log(2 * np.pi)
        - 0.5 * logdet_C_xt
        - 0.5 * np.sum(Y ** 2 * 1 / C_xt)
    )


def kron_loglike_ND_tridiag(
    y: Union[List, np.ndarray],
    x: List,
    std_meas: Union[int, float],
    std_model: Union[List, np.ndarray, int, float],
    lcorr_d: Union[int, float, np.ndarray],
    check_finite=False,
) -> float:
    """
    Args:
        y:
        x:
        std_meas:
        std_model:
        lcorr_d:
        check_finite:

    Returns:

    References:
        [1] Efficient inference in matrix-variate Gaussian models with
        i.i.d. observation noise. Stegle et al. (2011)

        [2] E Gilboa, Y Saatçi, JP Cunningham - Scaling Multidimensional
        Inference for Structured Gaussian Processes, Pattern Analysis and
        Machine Intelligence, IEEE, 2015
    """

    # Initialize arrays
    Nd = []
    lambda_d = []
    w_d = []
    C = 1

    # Loop over dimensions
    for i, d in enumerate(x):
        Nd.append(len(d))

        # Tridiagonal inverse covariance matrix and eigendecomposition
        d0, d1 = inv_cov_vec_1D(d, lcorr_d[i], std_model[i])
        lambda_i, w_i = eigh_tridiagonal(d0, d1, check_finite=check_finite)

        # Kronecker product of eigenvalues
        C = np.kron(1 / lambda_i, C)

        # Append to list
        lambda_d.append(lambda_i)
        w_d.append(w_i)

    # Add noise and calculate determinant
    C = C + std_meas ** 2
    logdet_C = np.sum(np.log(C))

    # Kronecker mvm. Note that eigenvec(A) = eigenvec(A^-1)
    a = kron_op(w_d, y, transa=True)
    a = a * 1 / C
    a = kron_op(w_d, a)
    ySy = np.sum(y * a)

    # Loglikelihood
    return -0.5 * np.prod(Nd) * np.log(2 * np.pi) - 0.5 * logdet_C - 0.5 * ySy


def chol_loglike_1D(
    coord_x: np.ndarray,
    x_model: np.ndarray,
    x_meas: np.ndarray,
    l_corr: Union[int, float],
    std_model: np.ndarray,
    std_meas: np.ndarray,
) -> float:
    """
    Efficient Gaussian loglikelihood for 1D problems with multiplicative
    exponential covariance.

    Linear time solution for 1D (e.g. timeseries) observations with gaussian
    i.i.d. white noise and multiplicative modeling uncertainty with exponential
    correlation.

    Args:
        coord_x: [N, ] vector of coordinates
        x_model: [N, ] vector of model predictions
        x_meas: [N, ] vector of measurements
        l_corr: Scalar correlation length
        std_model: [N, ] vector of model prediction uncertainty coefficient
        of variation
        std_meas: [N, ] vector of measurement uncertainty std. dev.

    Returns:
    """

    # Initialization
    Nx = len(coord_x)

    # Inverse covariance in vector form
    d0, d1 = inv_cov_vec_1D(coord_x, l_corr, std_model)

    # Assemble terms of Eqs 50 - 52
    W = 1 / (np.ones(Nx) * std_meas ** 2)  # Inverse noise vector
    yWy = np.sum(x_meas ** 2 * (1 / std_meas ** 2))  # Obtained from Woodbury id
    GWG = x_model ** 2 * (1 / std_meas ** 2)
    Wyx = W * x_model * x_meas
    d0_yWy = d0 + GWG

    # Factorize symmetric tridiagonal matrices
    D, L = chol_tridiag(d0, d1)
    Dw, Lw = chol_tridiag(d0_yWy, d1)

    # Cholesky solve
    X = cho_solve_symm_tridiag(Dw, Lw, Wyx)
    ySigmay = yWy - np.sum(Wyx * X)

    # Calculate determinants
    logdet_cov = -2 * np.sum(np.log(D))  # Logdet of the inverse covariance matrix
    logdet_C_yWy = 2 * np.sum(np.log(Dw))  # Logdet of the inverse cov + Y*W*Y
    logdet_W = np.sum(-np.log(W))
    logdet_Sigma = logdet_C_yWy + logdet_W + logdet_cov

    # Return loglikelihood
    return -0.5 * (logdet_Sigma + ySigmay + Nx * np.log(2 * np.pi))
