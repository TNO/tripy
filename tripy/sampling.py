import numpy as np
from numba import prange

from tripy.utils import chol_tridiag, inv_cov_vec_1D, kron_op, solve_lin_bidiag_mrhs


def chol_sample_1D(coords, std_noise, std_model, lcorr, y_model=None, size=1):
    """

    Args:
        coords:
        std_noise:
        std_model:
        lcorr:
        y_model:
        size:

    Returns:

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
    coords_x,
    coords_t,
    std_noise,
    std_model_x,
    std_model_t,
    lcorr_x,
    lcorr_t,
    y_model=None,
    size=1,
):

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


def kron_sample_ND(coords, std_noise, std_model, lcorr, y_model=None, size=1):
    """
    :param coords:
    :param std_noise:
    :param std_model:
    :param lcorr:
    :param y_model:
    :param size:
    :return:
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
