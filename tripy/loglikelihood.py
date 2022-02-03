"""
Likelihood evaluation
"""

from typing import Callable, List, Optional, Union

import numpy as np
import torch
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.linalg.lapack import dpotrf
from torch.distributions import MultivariateNormal
from torch.distributions.normal import Normal

from tripy.utils import (
    _cast_scalar_to_array,
    cho_solve_symm_tridiag,
    chol_tridiag,
    inv_cov_vec_1D,
    kron_op,
    symm_tri_block_chol,
    symm_tri_block_solve,
    symm_tri_mvm,
)


def chol_loglike_1D(
    y: np.ndarray,
    x: np.ndarray,
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
        y: [N, ] Vector of observations or residuals between measurements
        and model predictions.
        x: [N, ] vector of coordinates
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
        std_meas = _cast_scalar_to_array(std_meas, Nx)

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
        - At least one generalized dimension (space or time) has only one
        subdimension and exponential correlation

    Notes:
        * The tridiagonal inverse of the correlation matrix for exponential
        correlation can be obtained using `utils.inv_cov_vec_1D`.
        * Having the spatial correlation matrix as an input would result in a
        full inversion performed for each call to this function which is
        potentially unecessary. This is solved by specifying the spatial
        correlation in terms of the inverse.

    Warnings:
        * Directly inverting the correlation matrix to obtain `Cx` in the general
        case is computationally expensive and numerically unstable. To avoid numerical
        issues it is suggested to add a small value to the diagonal (jitter) before
        inverting. Values smaller than 1e-6 are typically sufficient.

    Args:
        y: [Nx, Nt] Array of observations or residuals between measurements
        and model predictions.
        Cx: List of diagonal and off diagonal vectors of inverse Exponential
        correlation matrix or full inverse of spatial correlation matrix.
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
            "Cx must be a numpy array containing the correlation"
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

    if (Nx < 2) or (Nt < 2):
        raise ValueError(
            f"The number of points in each dimension must be > 2"
            f"but is Nx = {Nx} and Nt = {Nt}."
        )

    # Cast noise to array
    std_meas = _cast_scalar_to_array(std_meas, (Nx, Nt))

    # Set y_model to vector of ones if None is specified
    if y_model is None:
        y_model = np.ones(Nx, Nt)

    L, C = symm_tri_block_chol(Cx, Ct, std_meas ** 2, y=y_model)

    # Get diagonal elements of L
    Ldiag = np.diagonal(L, axis1=1, axis2=2)

    # Vectors to be used later
    Winv_vec = 1 / std_meas ** 2
    yWy = np.sum(y ** 2 * (1 / std_meas ** 2))
    WGx = Winv_vec * y_model * y

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
    uncertainty and i.i.d. Gaussian noise.

    Warnings:
        * This is special case of kron_loglike_2D where both space
        and time have exponential correlation.
        * This is superceded by kron_loglike_ND_tridiag and will be
        removed.

    Args:
        y: [Nx, Nt] Array of observations or residuals between measurements
        and model predictions.
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

    # Kronecker prod of eigenvalues.
    if std_meas is not None:
        if not isinstance(std_meas, (int, float)):
            raise ValueError(f"`std_meas` must be {int}, {float} or {None}.")
        # Kronecker prod of eigenvalues.
        C_xt = np.kron(1 / lambda_x, 1 / lambda_t) + std_meas ** 2
    else:
        C_xt = np.kron(1 / lambda_x, 1 / lambda_t)

    # Determinant: C_xt is diagonal. We can therefore sum the log terms.
    # This is the determinant of the inverse.
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
    Cx: Union[list, np.ndarray],
    Ct: Union[list, np.ndarray],
    std_meas: Optional[Union[int, float]],
    check_finite: Optional[bool] = False,
) -> float:
    """
    Efficient loglikelihood for Exponential temporal correlation.

    Any combination of a single dimension with Markovian covariance and N
    nonseparable dimensions with non-Markovian covariance can be evaluated
    using this function.

    Notes:
        * The tridiagonal inverse of the correlation matrix for exponential
        correlation can be obtained using `utils.inv_cov_vec_1D`.

    Args:
        y: [Nx, Nt] Array of observations or residuals between measurements
        and model predictions.
        std_meas: Std. dev. of measurement uncertainty.
        Cx: Correlation matrix of the model prediction uncertainty in space, or
        list with diagonal and off-diagonal of tridiagonal inverse.
        Ct: Correlation matrix of the model prediction uncertainty in time, or
        list with diagonal and off-diagonal of tridiagonal inverse.
        check_finite: Optional flag of scipy.linalg.eigh_tridiagonal.

    Returns:
        L: Loglikelihood.
    """

    # Check if the supplied Cx and Ct are correlation matrices or lists of
    # the diagonal and off-diagonal vectors of a tridiagonal inverse correlation
    # matrix. Apply the corresponding eigendecomposition.
    if isinstance(Cx, list):
        Nx = np.shape(Cx[0])
        lambda_x, w_x = eigh_tridiagonal(Cx[0], Cx[1], check_finite=check_finite)
        lambda_x = 1 / lambda_x
    elif isinstance(Cx, np.ndarray):
        Nx = np.shape(Cx)[0]
        lambda_x, w_x = eigh(Cx, check_finite=check_finite)
    else:
        raise ValueError(
            "Cx must be a numpy array containing the correlation"
            "matrix, or a list of the diagonal and off-diagonal"
            "vectors of the inverse correlation matrix."
        )

    if isinstance(Ct, list):
        Nt = np.shape(Ct[0])
        lambda_t, w_t = eigh_tridiagonal(Ct[0], Ct[1], check_finite=check_finite)
        lambda_t = 1 / lambda_t
    elif isinstance(Ct, np.ndarray):
        Nt = np.shape(Ct)[0]
        lambda_t, w_t = eigh(Ct, check_finite=check_finite)
    else:
        raise ValueError(
            "Ct must be a numpy array containing the correlation"
            "matrix, or a list of the diagonal and off-diagonal"
            "vectors of the inverse correlation matrix."
        )

    if std_meas is not None:
        if not isinstance(std_meas, (int, float)):
            raise ValueError(f"`std_meas` must be {int}, {float} or {None}.")

        # Kronecker prod of eigenvalues.
        C_xt = np.kron(lambda_x, lambda_t) + std_meas ** 2

    else:
        C_xt = np.kron(lambda_x, lambda_t)

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
    std_model: Union[List, np.ndarray],
    l_corr_d: Union[List, np.ndarray],
    std_meas: Optional[Union[int, float]],
    check_finite: Optional[bool] = False,
) -> float:
    """
    Args:
        y: Vector of observations or residuals between measurements
        and model predictions.
        x: List of lists, each containing the coordinate vector of a dim.
        std_meas: Std. dev. of the measurement uncertainty.
        std_model: List of vectors of the model uncertainty std. dev. per dim.
        l_corr_d: Correlation lengths for each dimension.
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
        d0, d1 = inv_cov_vec_1D(d, l_corr_d[i], std_model[i])
        lambda_i, w_i = eigh_tridiagonal(d0, d1, check_finite=check_finite)

        # Kronecker product of eigenvalues
        C = np.kron(1 / lambda_i, C)

        # Append to list
        lambda_d.append(lambda_i)
        w_d.append(w_i)

    # Add noise and calculate determinant
    if std_meas is not None:
        if not isinstance(std_meas, (int, float)):
            raise ValueError(f"`std_meas` must be {int}, {float} or {None}.")
        C = C + std_meas ** 2
    logdet_C = np.sum(np.log(C))

    # Kronecker mvm. Note that eigenvec(A) = eigenvec(A^-1)
    a = kron_op(w_d, y, transA=True)
    a = a * 1 / C
    a = kron_op(w_d, a)
    ySy = np.sum(y * a)

    # Loglikelihood
    return -0.5 * np.prod(Nd) * np.log(2 * np.pi) - 0.5 * logdet_C - 0.5 * ySy


def log_likelihood_linear_normal(
    y: np.ndarray,
    K: Optional[Union[int, float, np.ndarray]],
    std_meas: Optional[Union[int, float, np.ndarray]] = None,
    y_model: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Gaussian log-likelihood function for the following models:

    ``X_model = K1 * f(theta) + E``
    ``X_model = K2 + E``

    where
        * ``K1`` ~ MVN(mean=1, covariance=k_cov_mx)
        * ``K2`` ~ MVN(mean)
        * ``E`` ~ MVN(mean=0, covariance=e_cov_mx)
        * ``f`` is a physical model function parametrized by theta

    Notes:
        * If K is None, the loglikelihood is calculated as the sum of N
        independent normally distributed random variables

    TODO:
        * The logic in this function works but can probably be improved.

    Args:
        y: [N_1, N_2, ... Ni, ... N_d] Array of observations or residuals
        between measurements and model predictions.
        K: covariance matrix of `K`, shape [prod(Ni), prod(Ni)].
        std_meas: [shape(y)] Std. dev. of the measurement uncertainty with
        y_model: [shape(y)] optional vector of model predictions in the case of
        multiplicative model prediction uncertainty.

    Returns:
        log_like: value of the log-likelihood evaluated at theta.

    """
    # Pre-process
    y = y.ravel()
    N = y.size
    y = torch.tensor(y)

    # Check that model uncertainty covariance matrix is supplied if `y_model`
    # is not None. Scale model uncertainty covariance by model prediction.
    if y_model is not None:
        if K is not None:
            y_model_diag = np.diag(y_model.ravel())
            K = np.matmul(y_model_diag, np.matmul(K, y_model_diag))
        else:
            raise ValueError(
                "Model uncertainty scaling vector is provided but `K` is None"
            )

    # If K is not supplied, then the loglikelihood can be evaluated as the
    # sum of loglikelihoods of independent normally distributed random vars
    if K is None:
        if std_meas is not None:
            std_meas = _cast_scalar_to_array(std_meas, N).ravel()
            dist = Normal(torch.zeros(N), torch.tensor(std_meas))
        else:
            raise ValueError("At least one of `K`, `std_meas` expected to be not None")

    # Multivariate normal case
    else:
        if std_meas is not None:
            std_meas = _cast_scalar_to_array(std_meas, N).ravel()
            dist = MultivariateNormal(
                torch.zeros(N), torch.tensor(K + np.diag(std_meas ** 2))
            )
        else:
            dist = MultivariateNormal(torch.zeros(N), torch.tensor(K))

    return dist.log_prob(y).sum().detach().cpu().numpy()


def log_likelihood_reference(
    theta: np.ndarray,
    x_meas: np.ndarray,
    physical_model: Callable[[np.ndarray], np.ndarray],
    k_cov_mx: Optional[np.ndarray] = None,
    e_cov_mx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reference log-likelihood function for the following model,
    for testing:

    ``X_model = K*physical_model_fun(theta) + E``

    where
        * ``K`` ~ MVN(mean=1, covariance=K)
        * ``E`` ~ MVN(mean=0, covariance=e_cov_mx)
        * ``MVN()`` multivariate normal distribution

    ``physical_model_fun()`` has no additional uncertainty, e.g. surrogate model
    uncertainty.

    Notes:
        * This function is taken from `taralli` (https://gitlab.com/tno-bim/taralli/)

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
