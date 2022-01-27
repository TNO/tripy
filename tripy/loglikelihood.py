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
    _cast_to_array,
    cho_solve_symm_tridiag,
    chol_tridiag,
    inv_cov_vec_1D,
    kron_op,
    symm_tri_block_chol,
    symm_tri_block_solve,
    symm_tri_mvm,
)


def chol_loglike_2D(
    y: np.ndarray,
    Cx: Union[np.ndarray, List],
    Ct: List,
    std_meas: Union[int, float, np.ndarray],
    y_model: Optional[np.ndarray] = None,
) -> float:
    """
    Efficient 2D loglikelihood using block Cholesky decomposition.

    Given the uncertainty parameters and vectors of the x and t positions
    on a lattice, calculates the loglikelihood of the 2D Gaussian process
    with multiplicative space and time uncertainties and i.i.d. noise. It
    is intended that this function is general and works with any combination
    of spatial and temporal covariance, with only the following restrictions:
        - Space and time are separable (Kronecker structure)
        - Correlation in time is exponential

    Args:
        y: [Nx, Nt] Array of observations.
        Cx: List of diagonal and off diagonal or full inverse of spatial
            correlation matrix.
        Ct: List of iagonal and off-diagonal of inverse of Exponential temporal
         correlation matrix.
        std_meas: Std. dev. of the measurement uncertainty.
        y_model: [Nx, Nt] optional vector of model predictions in the case of
        multiplicative model prediction uncertainty.

    Returns:
        L: Loglikelihood.
    """

    # Check if Cx and Ct are lists or numpy arrays.
    if isinstance(Cx, list):
        # TODO: Change dpttrf to chol_tridiag
        Nx = len(Cx[0])
        Lx = chol_tridiag(Cx[0], Cx[1])[0]
    elif isinstance(Cx, np.ndarray):
        Nx = np.shape(Cx)[0]
        Lx = np.diag(dpotrf(Cx, lower=1, clean=1, overwrite_a=0)[0])
    else:
        raise ValueError(
            "C_x must be a numpy array containing the correlation"
            "matrix, or a list of the diagonal and off-diagonal"
            "vectors of the inverse correlation matrix."
        )

    if isinstance(Ct, list):
        Nt = len(Ct[0])
        Lt = chol_tridiag(Ct[0], Ct[1])[0]
    else:
        raise ValueError(
            "Ct must be a list containing the diagonal and "
            "off-diagonal vectors of a tridiagonal matrix"
        )

    # Set y_model to vector of ones if None is specified
    if y_model is None:
        y_model = np.ones(Nt, Nx)

    L, C = symm_tri_block_chol(Cx, Ct, std_meas ** 2, y=y_model)

    # Get diagonal elements of L
    Ldiag = np.diagonal(L, axis1=1, axis2=2)

    # Cast noise to array
    std_meas = _cast_to_array(std_meas, Nx * Nt)

    # Vectors to be used later
    Winv_vec = 1 / std_meas ** 2
    yWy = np.sum(y ** 2 * (1 / std_meas ** 2))
    WGx = Winv_vec * y_model.ravel() * y.ravel()

    # Solve the linear system
    X = symm_tri_block_solve(L, C, WGx, Nx, Nt)

    # Vector product
    xSx = yWy - np.sum(WGx * X)

    # ========================================================================
    # Logdeterminant
    # ========================================================================

    # Logdet of noise matrix
    logdet_W = np.sum(2 * np.log(std_meas))

    # Logdet of the full correlation matrix and Cholesky factors
    logdet_C = -2 * np.sum(np.log(Lx)) * Nt + -2 * np.sum(np.log(Lt)) * Nx
    logdet_chol = 2 * np.sum(np.log(Ldiag))

    # Sum logdeterminants
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
        physical_model: function that takes as input theta and gives out y_model.
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
    y: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    std_meas: Union[int, float],
    l_corr_x: Union[int, float],
    std_x: Union[int, float],
    l_corr_t: Union[int, float],
    std_t: Union[int, float],
    check_finite: Optional[bool] = False,
) -> float:
    """
    Efficient 2D loglikelihood using Kronecker properties.

    Given the uncertainty parameters and vectors of the x and t positions
    on a lattice, calculates the loglikelihood of the observations
    for exponential space and time correlation in the additive modeling
    uncertainty and i.i.d. Gaussian noise

    Warnings:
        * This is superceded by kron_loglike_ND_tridiag and will be
        removed.

    Args:
        y: [Nx, Nt] Array of observations.
        x: [Nx,] Vector of space coordinates.
        t: [Nt,] Vector of time coordinates.
        std_meas: Vector of measurement uncertainty std. dev.
        l_corr_x: Spatial correlation length.
        std_x: Std. dev. of model prediction uncertainty in space.
        l_corr_t: Temporal correlation length.
        std_t: Std. dev. of model prediction uncertainty in time.
        check_finite: Optional flag of scipy.linalg.eigh_tridiagonal.

    Returns:
        L: Loglikelihood.
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


def kron_loglike_2D(
    y: np.ndarray,
    C_x: Union[list, np.ndarray],
    C_t: Union[list, np.ndarray],
    std_meas: Optional[Union[int, float]],
    check_finite: Optional[bool] = False,
) -> float:
    """
    Efficient loglikelihood for Exponential temporal correlation.

    Any combination of a single dimension with Markovian covariance and N
    nonseparable dimensions with non-Markovian covariance can be evaluated
    using this function.

    Args:
        y: [Nx, Nt] Array of observations.
        std_meas: Std. dev. of measurement uncertainty.
        C_x: Correlation matrix of the model prediction uncertainty in space, or
        list with diagonal and off-diagonal of tridiagonal inverse.
        C_t: Correlation matrix of the model prediction uncertainty in time, or
        list with diagonal and off-diagonal of tridiagonal inverse.
        check_finite: Optional flag of scipy.linalg.eigh_tridiagonal.

    Returns:
        L: Loglikelihood.
    """

    # Check if the supplied C_x and C_t are correlation matrices or lists of
    # the diagonal and off-diagonal vectors of a tridiagonal inverse correlation
    # matrix. Apply the corresponding eigendecomposition.
    if isinstance(C_x, list):
        Nx = np.shape(C_x[0])
        lambda_x, w_x = eigh_tridiagonal(C_x[0], C_x[1], check_finite=check_finite)
        lambda_x = 1 / lambda_x
    elif isinstance(C_x, np.ndarray):
        Nx = np.shape(C_x)[0]
        lambda_x, w_x = eigh(C_x, check_finite=check_finite)
    else:
        raise ValueError(
            "C_x must be a numpy array containing the correlation"
            "matrix, or a list of the diagonal and off-diagonal"
            "vectors of the inverse correlation matrix."
        )

    if isinstance(C_t, list):
        Nt = np.shape(C_t[0])
        lambda_t, w_t = eigh_tridiagonal(C_t[0], C_t[1], check_finite=check_finite)
        lambda_t = 1 / lambda_t
    elif isinstance(C_t, np.ndarray):
        Nt = np.shape(C_t)[0]
        lambda_t, w_t = eigh(C_t, check_finite=check_finite)
    else:
        raise ValueError(
            "C_t must be a numpy array containing the correlation"
            "matrix, or a list of the diagonal and off-diagonal"
            "vectors of the inverse correlation matrix."
        )

    # Kronecker prod of eigenvalues.
    C_xt = np.kron(lambda_x, lambda_t) + std_meas ** 2

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
    y: np.ndarray,
    x: List,
    std_meas: Union[int, float],
    std_model: Union[List, np.ndarray],
    lcorr_d: Union[List, np.ndarray],
    check_finite: Optional[bool] = False,
) -> float:
    """
    Args:
        y: Vector of measurements.
        x: List of lists, each containing the coordinate vector of a dim.
        std_meas: Std. dev. of the measurement uncertainty.
        std_model: List of vectors of the model uncertainty std. dev. per dim.
        lcorr_d: Correlation lengths for each dimension.
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
    a = kron_op(w_d, y, transA=True)
    a = a * 1 / C
    a = kron_op(w_d, a)
    ySy = np.sum(y * a)

    # Loglikelihood
    return -0.5 * np.prod(Nd) * np.log(2 * np.pi) - 0.5 * logdet_C - 0.5 * ySy


def chol_loglike_1D(
    x: np.ndarray,
    y: np.ndarray,
    l_corr: Union[int, float],
    std_model: Union[int, float, np.ndarray],
    std_meas: Optional[Union[int, float, np.ndarray]] = None,
    y_model: Optional[np.ndarray] = None,
) -> float:
    """
    Efficient Gaussian loglikelihood for 1D problems with exponential correlation.

    Linear time solution for 1D (e.g. timeseries) observations with additive
    Gaussian noise and exponential model prediction uncertainty. The model
    prediction uncertainty can be multiplicative or additive.

    Args:
        x: [N, ] vector of coordinates
        y: [N, ] vector of residuals between measurement and model prediction.
        l_corr: Scalar correlation length
        std_model: [N, ] vector of model prediction uncertainty coefficient
        of variation in case of multiplicative model prediction uncertainty,
        or vector of std. devs. for additive in the case of additive model
        prediction uncertainty.
        std_meas: [N, ] optional vector of measurement uncertainty std. dev.
        y_model: [N, ] optional vector of model predictions in the case of
        multiplicative model prediction uncertainty.

    Returns:
        L: Loglikelihood.
    """

    # Initialization
    Nx = len(x)

    # Set y_model to vector of ones if None is specified
    if y_model is None:
        y_model = np.ones(Nx)

    if std_meas is None:
        # Inverse covariance in vector form
        d0, d1 = inv_cov_vec_1D(x, l_corr, std_model * y_model)

        # Factorize
        D, L = chol_tridiag(d0, d1)

        # Cholesky solve
        X = symm_tri_mvm(d0, d1, y)
        ySigmay = np.sum(y * X)

        # Determinants
        logdet_Sigma = -2 * np.sum(np.log(D))

        # Sum terms
        loglike = -0.5 * (logdet_Sigma + ySigmay + Nx * np.log(2 * np.pi))

    else:
        # Cast noise std. dev. to array
        std_meas = _cast_to_array(std_meas, Nx)

        # Inverse covariance in vector form
        d0, d1 = inv_cov_vec_1D(x, l_corr, std_model)

        # Assemble terms of Eqs 50 - 52
        W = 1 / (np.ones(Nx) * std_meas ** 2)  # Inverse noise vector
        yWy = np.sum(y ** 2 * (1 / std_meas ** 2))  # Obtained from Woodbury id
        GWG = y_model ** 2 * (1 / std_meas ** 2)
        Wyx = W * y_model * y
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

        # Sum terms
        loglike = -0.5 * (logdet_Sigma + ySigmay + Nx * np.log(2 * np.pi))

    return loglike
