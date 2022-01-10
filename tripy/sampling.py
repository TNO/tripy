from typing import List, Optional, Union

import numpy as np
from numba import prange

from tripy.utils import chol_tridiag, inv_cov_vec_1D, kron_op, solve_lin_bidiag_mrhs


def chol_sample_1D(
    coords: np.ndarray,
    std_noise: Union[np.ndarray, float, int],
    std_model: np.ndarray,
    lcorr: Union[float, int],
    y_model: Optional[np.ndarray] = None,
    size: Optional[int] = 1,
) -> np.ndarray:
    """
    Efficient sampling from Exponentially correlated MVN distribution in 1D.

    Utilizes the tridiagonal inverse of the correlation matrix to efficiently
    sample from large Multivariate Normal distributions using a Cholesky decomposition.

    Warnings:
        This function has not been validated against a reference implementation, and
        should not be trusted.

    Args:
        coords: [1, N] vector of coordinates.
        std_noise: [1, N] vector of std. dev. of the measurement uncertainty.
        std_model: [1, N] vector of std. dev. of the modeling uncertainty.
        lcorr: Scalar correlation length.
        y_model: [1, N] vector of model predictions.
        size: Number of samples.

    Returns:
        [size, N] array of samples.
    """

    N = len(coords)

    # Sample from std. normal and Gaussian with vector noise
    X = np.random.default_rng().normal(loc=0.0, scale=1.0, size=(size, N))
    S = np.random.default_rng().normal(loc=0.0, scale=std_noise, size=(size, N))

    # Cholesky of inverse correlation matrix
    d0, d1 = inv_cov_vec_1D(coords, lcorr, std_model)
    l0, l1 = chol_tridiag(d0, d1)

    # Solve linear bidiagonal system
    Z = solve_lin_bidiag_mrhs(l0, l1, X.T, side="U")

    # Scale by model output
    if y_model is not None:
        Z = y_model * Z.T

    # Sum the samples
    return Z + S


def kron_sample_2D(
    coords_x: np.ndarray,
    coords_t: np.ndarray,
    std_noise: Union[np.ndarray, float, int],
    std_model_x: np.ndarray,
    std_model_t: np.ndarray,
    lcorr_x: Union[float, int],
    lcorr_t: Union[float, int],
    y_model: Optional[np.ndarray] = None,
    size: Optional[int] = 1,
) -> np.ndarray:
    """
    Efficient sampling from Exponentially correlated MVN distribution in 2D.

    Utilizes the tridiagonal inverse of the correlation matrix to efficiently
    sample from large Multivariate Normal distributions using Kronecker properties.

    Warnings:
        This function has not been validated against a reference implementation, and
        should not be trusted.

    Args:
        coords_x: [1, Nx] vector of spatial coordinates.
        coords_t: [1, Nt] vector of temporal coordinates.
        std_noise: [1, Nx * Nt] Vector std. dev. of the measurement uncertainty.
        std_model_x: [1, Nx] Vector std. dev. of the modeling uncertainty in space.
        std_model_t: [1, Nt] Vector std. dev. of the modeling uncertainty in time.
        lcorr_x: Scalar spatial correlation length.
        lcorr_t: Scalar temporal correlation length.
        y_model: [Nx, Nt] Array of model predictions.
        size: Scalar number of samples.

    Returns:
        [size, Nx * Nt] array of samples.
    """

    # Get size of field
    Nx = len(coords_x)
    Nt = len(coords_t)

    # Get diagonals and off-diagonals and factorize
    d0_x, d1_x = inv_cov_vec_1D(coords_x, lcorr_x, std_model_x)
    d0_t, d1_t = inv_cov_vec_1D(coords_t, lcorr_t, std_model_t)
    Lx = chol_tridiag(d0_x, d1_x)
    Lt = chol_tridiag(d0_t, d1_t)

    # Sample from std. normal and Gaussian with vector noise
    X = np.random.default_rng().normal(loc=0.0, scale=1.0, size=(size, Nx * Nt))
    S = np.random.default_rng().normal(loc=0.0, scale=std_noise, size=(size, Nx * Nt))

    # Solve for Z
    def solve_func(A, B):
        return solve_lin_bidiag_mrhs(A[0], A[1], B, side="U")

    Z = np.zeros((size, (Nx * Nt)))
    for i in prange(size):
        Z[i, :] = kron_op([Lx, Lt], X[i, :], solve_func)

    # Scale by model output
    if y_model is not None:
        Z = np.ravel(y_model) * Z

    # Sum the samples
    return Z + S


def kron_sample_ND(
    coords: List,
    std_noise: np.ndarray,
    std_model: List,
    lcorr: np.ndarray,
    y_model: Optional[np.ndarray] = None,
    size: Optional[int] = 1,
) -> np.ndarray:
    """
    Efficient sampling from Exponentially correlated MVN distribution in ND.

    Utilizes the tridiagonal inverse of the correlation matrix to efficiently
    sample from large Multivariate Normal distributions using Kronecker properties.

    Warnings:
        This function has not been validated against a reference implementation, and
        should not be trusted.

    Args:
        coords: [ND] list of vectors of coordinates per dimension.
        std_noise: [1, prod(ND)] vector of the std. dev. of the measurement uncertainty.
        std_model: [ND] list of vectors of the modeling uncertainty std. devs. per dim.
        lcorr: [ND, 1] vector of correlation lengths per dim.
        y_model: [N1 x N2 x ... x ND] array of model predictions.
        size: Scalar number of samples.

    Returns:
        [size, prod(ND)] array of samples.
    """
    # coords: List of ND vectors
    # std_noise: Vector of size prod(ND)
    # std_model: List of ND vectors
    # lcorr: Vector of size ND
    # y_model: Array of size [N1 x N2 x ... x ND]

    Ni = []
    L = []
    for idx_N, coords_i in enumerate(coords):
        Ni.append(len(coords_i))
        d0_i, d1_i = inv_cov_vec_1D(coords_i, lcorr[idx_N], std_model[idx_N])
        Li = chol_tridiag(d0_i, d1_i)
        L.append(Li)

    # Number of points
    Npts = np.prod(Ni)

    # Sample from std. normal and Gaussian with vector noise
    X = np.random.default_rng().normal(loc=0.0, scale=1.0, size=(size, Npts))
    S = np.random.default_rng().normal(loc=0.0, scale=std_noise, size=(size, Npts))

    # Solve for Z across all dimensions
    def solve_func(A, B):
        return solve_lin_bidiag_mrhs(A[0], A[1], B, side="U")

    Z = np.zeros((size, Npts))
    for i in prange(size):
        Z[i, :] = kron_op(L, X[i, :], solve_func)

    # Scale by model output
    # TODO: Check that ravel order matches `kron_op`
    if y_model is not None:
        Z = np.ravel(y_model, order="C") * Z

    # Sum the samples
    return Z + S
