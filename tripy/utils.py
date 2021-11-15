"""
 Utility functions for the log-likelihood builder module.
"""
from itertools import product
from timeit import default_timer as timer
from typing import Callable, Tuple, Union

import numpy as np
from numba import jit, prange
from scipy.linalg.blas import dgemm, dsyrk, dtrsm
from scipy.linalg.lapack import dpotrf, dpttrf
from scipy.spatial.distance import pdist, squareform


def correlation_matrix(
    x_mx,
    correlation_func: Callable[[np.ndarray], np.ndarray],
    distance_metric: str = "euclidean",
) -> np.ndarray:
    """
    Statistical correlation matrix between measurement points based on their distance.

    Args:
        x_mx: the coordinates of the measurement points between which the correlation
            matrix is calculated, (N,K).
        correlation_func: a function that calculates the statistical correlation
            between measurements made at two points that are `d` units distant from
            each other, r_major^m -> r_major^m.
        distance_metric: The distance metric to use. The distance function is the
            same as the `metric` input argument of `scipy.spatial.distance.pdist`.

    Returns:
        rho_mx: correlation matrix.
    """

    d = squareform(pdist(x_mx, metric=distance_metric))
    rho_mx = correlation_func(d)

    return rho_mx


def correlation_function(
    d: Union[np.ndarray, list, float, int],
    correlation_length: Union[float, int],
    function_type: str = "exponential",
    exponent: Union[float, int] = 2,
) -> np.ndarray:
    """
    Statistical correlation between measurements made at two points that are at `d`
    units distance from each other.

    Args:
        d: distance(s) between points.
        correlation_length: `1/correlation_length` controls the strength of
            correlation between two points. `correlation_length = 0` => complete
            independence for all point pairs (`rho=0`). `correlation_length = Inf` =>
            full dependence for all point pairs (`rho=1`).
        function_type: name of the correlation function.
        exponent: exponent in the type="cauchy" function. The larger the exponent the
            less correlated two points are.

    Returns:
        rho: correlation coefficient for each element of `d`.
    """
    function_type = function_type.lower()

    if correlation_length < 0:
        raise ValueError("correlation_length must be a non-negative number.")
    if np.any(d < 0):
        raise ValueError("All elements of d must be non-negative numbers.")

    if correlation_length == 0:
        idx = d == 0
        rho = np.zeros(d.shape)
        rho[idx] = 1
    elif correlation_length == np.inf:
        rho = np.ones(d.shape)
    else:
        if function_type == "exponential":
            rho = np.exp(-d / correlation_length)
        elif function_type == "cauchy":
            rho = (1 + (d / correlation_length) ** 2) ** -exponent
        elif function_type == "gaussian":
            rho = np.exp(-((d / correlation_length) ** 2))
        else:
            raise ValueError(
                "Unknown function_type. It must be one of these: 'exponential', "
                "'cauchy', 'gaussian'."
            )
    return rho


def grow_mx(seed_mx: np.ndarray, growth_scale: int) -> np.ndarray:
    """
    Grow a 2D array while keeping its original "look".

    Args:
        seed_mx: 2D-array to be grown
        growth_scale:

    Returns:
        growth_mx
    """
    grown_list = [
        [np.ones((growth_scale, growth_scale)) * elem for elem in row]
        for row in seed_mx
    ]
    grown_mx = np.block(grown_list)
    return grown_mx


# @jit(nopython=True)
def symm_tri_block_chol(Cx, Ct, vec_diag, y=None):
    """
    Block Cholesky decomposition for symmetric block tridiagonal matrices

    The input matrix is assumed to be the kronecker product of a symmetric
    tridiagonal matrix Ct and a general matrix Cx.

    References:
        https://software.intel.com/content/www/us/en/develop/documentation/onemkl-cookbook/
        top/factoring-block-tridiagonal-symmetric-positive-definite-matrices.html
    """
    # ========================================================================
    # Initialization
    # ========================================================================

    # Check if Cx and Ct are lists or numpy arrays.
    if type(Cx) == list:
        Nx = len(Cx[0])
        Dd0_u = np.diag(Cx[1], k=1)
        Dd0_l = np.diag(Cx[1], k=-1)
        Dd0 = Dd0_u + Dd0_l
        np.fill_diagonal(Dd0, Cx[0])
    else:
        Nx = np.shape(Cx)[0]
        Dd0 = Cx

    Nt = len(Ct[0])

    # If y is not supplied create an array of ones
    if y is None:
        y = np.ones((Nt, Nx))
    else:
        y = np.reshape(y, (Nt, Nx))

    # Reshape diagonal vector into array
    vec_diag = np.reshape(vec_diag, (Nt, Nx))

    # Product of noise and model output matrices
    GWG = y ** 2 / vec_diag

    # Cholesky of space covariance matrix
    Li = dpotrf(Dd0 * Ct[0][0] + np.diag(GWG[0]), lower=1, clean=1, overwrite_a=0)[0]
    # Li = np.linalg.cholesky(Dd0 * Ct[0][0] + np.diag(GWG[0]))

    # Loop over blocks
    L = []
    C = []
    for i in range(Nt - 1):
        # DTRSM
        L.append(Li)
        Bi = Dd0 * Ct[1][i]
        Ci = dtrsm(1, Li, Bi, side=1, lower=1, trans_a=1, diag=0, overwrite_b=0)
        C.append(Ci)

        # DSYRK
        Di = Dd0 * Ct[0][i + 1] + np.diag(GWG[i + 1])
        Di = dsyrk(-1.0, Ci, beta=1.0, c=Di, trans=0, lower=1, overwrite_c=0)

        # Cholesky of space covariance matrix
        Li = dpotrf(Di, lower=1, clean=1, overwrite_a=0)[0]
    L.append(Li)

    return L, C


def symm_tri_block_solve(L, C, RHS, Nx, Nt):
    """
    Args:
        L:
        RHS:
        Nx:
        Nt:

    Returns:

    References: https://software.intel.com/content/www/us/en/develop/documentation/
    /mkl-cookbook/top/solve-lin-equations-block-tridag-symm-pos-definite-coeff-matrix.html

    """

    # DTRSM
    Yi = dtrsm(1, L[0], RHS[0:Nx], side=0, lower=1, trans_a=0, diag=0, overwrite_b=0)
    G = []
    Y = [Yi]
    for i in range(Nt - 1):
        # DGEMM
        Gi = dgemm(
            -1.0,
            C[i],
            Yi,
            beta=1.0,
            c=RHS[Nx * (i + 1) : Nx * (i + 2)],
            trans_a=0,
            trans_b=0,
            overwrite_c=0,
        )
        G.append(Gi)

        # DTRSM
        Yi = dtrsm(1, L[i + 1], Gi, side=0, lower=1, trans_a=0, diag=0, overwrite_b=0)

        Y.append(Yi)

    # DTRSM
    X = np.zeros((Nt, Nx))
    Xi = dtrsm(1.0, L[-1], Yi, side=0, lower=1, trans_a=1, diag=0, overwrite_b=0)
    X[-1] = Xi

    for i in range(Nt - 1, 0, -1):
        # DGEMM
        Zi = dgemm(
            -1.0,
            C[i - 1],
            Xi,
            beta=1.0,
            c=Y[i - 1],
            trans_a=1,
            trans_b=0,
            overwrite_c=0,
        )

        # DTRSM
        Xi = dtrsm(1.0, L[i - 1], Zi, side=0, lower=1, trans_a=1, diag=0, overwrite_b=0)
        X[i - 1] = Xi
    X = X.ravel()
    return X


def inv_cov_vec_1D(
    coord_x: np.ndarray, l_corr: Union[float, int], std: Union[np.ndarray, float, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates diagonal and off diagonal vectors of the tridiagonal inverse exponential
    covariance matrix

    Utility function for 2D loglikelihood evaluation. Given vectors of space
    and time points and the exponential covariance parameters returns the
    diagonal and off diagonal vectors of the tridiagonal inverse space and time
    covariance matrices. The expressions in [1] are modified to account for
    vector standard deviations.

    Args:
        coord_x: Vector of x
        l_corr: Correlation length in x
        std: Modeling uncertainty std dev or c.o.v. in x

    Returns:
        C_0: Main diagonal vector of inverse covariance matrix
        C_1: Off diagonal vector of inverse covariance matrix

    References:
        [1] Parameter Estimation for the Spatial Ornstein-Uhlenbeck
        Process with Missing Observations
        https://dc.uwm.edu/cgi/viewcontent.cgi?article=2131&context=etd
    """

    Nx = len(coord_x)
    a = np.exp(-np.diff(coord_x) / l_corr)

    # Initialize arrays
    C_0 = np.zeros(Nx)

    # Diagonal and off diagonal elements
    a11 = (1 / (1 - a[0] ** 2)) / std[0] ** 2
    ann = (1 / (1 - a[-1] ** 2)) / std[-1] ** 2
    aii = (1 / (1 - a[:-1] ** 2) + 1 / (1 - a[1:] ** 2) - 1) / std[1:-1] ** 2

    # Assemble the diagonal vectors
    C_0[0] = a11
    C_0[1:-1] = aii
    C_0[-1] = ann
    C_1 = (-a / (1 - a ** 2)) / (std[:-1] * std[1:])

    return C_0, C_1


def kron_op(
    A: list, b: np.ndarray, op_func: Callable = np.matmul, transA=False
) -> np.ndarray:
    """
    Efficient matrix-vector product or LSoE solve for Kronecker matrices

    The operation carried out by this function depends on the supplied `op_func`:

    * If `op_func` is a function for matrix-vector multiplication:
    Evaluate the product x = Ab

    * If `op_func` is a function for solving linear systems:
    Solve the linear system Ax = b

    where A is the Kronecker product of D matrices A = kron(AD, ..., A2, A1).
    The complexity of the opration depends on the supplied `op_func` and the types
    of matrices involved. For matrix-vector multiplication in the general case
    the theoretical complexity is  O(DN^[(D+1)/D]).

    Based on [1] and extended to solve linear systems.

    Args:
        A: List of D arrays where D is the number of dimensions.
        b: Vector of size N
        op_func: Callable function with call signature `op_func(A, X)`
        transA: Optional, transpose the arrays in A

    Returns:
        x: Vector of size N

    References:
        [1] E Gilboa, Y Saatçi, JP Cunningham - Scaling Multidimensional Inference
         for Structured Gaussian Processes, Pattern Analysis and Machine
         Intelligence, IEEE, 2015

    Notes:
        * The performance impact of the loop should be negligible, since the
        number of dimensions will be =< 4. Numba could be used to improve
        performance if necessary
        * TODO: Add a mvm_func for the case of diagonal A matrices.
    """
    N = len(b)
    for Ai in A:
        Gd = np.shape(Ai[0])[0]
        X = np.reshape(b, (Gd, int(N / Gd)))
        if transA:
            Z = op_func(Ai.T, X)
        else:
            Z = op_func(Ai, X)
        b = Z.T.ravel(order="C")
    return b.ravel(order="C")


def chol_tridiag(d0, d1):
    """
    Efficient symmetric tridiagonal cholesky

    Convenience function that converts the tridiagonal factorization
    A = LDL^T obtained from lapack.dpttrf into a Cholesky decomposition.

    Args:
        d0: Diagonal vector of symmetric tridiagonal matrix
        d1: Off-diagonal vector of symmetric tridiagonal matrix

    Returns:
        Diagonal and off-diagonal vector of L where A = LL^T
    """

    D, L, _ = dpttrf(d0, d1)
    D = np.sqrt(D)
    return D, D[:-1] * L


@jit(nopython=True)
def solve_lin_bidiag(d0, d1, b, side="L"):
    """
    Solve linear system with bidiagonal coefficient matrix
    """
    N = len(b)
    x = np.zeros(N)

    if side == "L":
        x[0] = b[0] / d0[0]
        for i in range(N - 1):
            x[i + 1] = (1 / d0[i + 1]) * (b[i + 1] - d1[i] * x[i])
    elif side == "U":
        x[-1] = b[-1] / d0[-1]
        for i in range(N - 1, 0, -1):
            x[i - 1] = 1 / d0[i - 1] * (b[i - 1] - d1[i - 1] * x[i])
    return x


@jit(nopython=True)
def solve_lin_bidiag_mrhs(d0, d1, B, side="L"):
    """
    Solve linear system with bidiagonal coefficient matrix and multiple RHS
    """
    N, n_rhs = np.shape(B)
    X = np.zeros((N, n_rhs))

    for i in range(n_rhs):
        X[:, i] = solve_lin_bidiag(d0, d1, B[:, i], side=side)
    return X


@jit(nopython=True)
def cho_solve_symm_tridiag(D, L, b):
    """
    Solve linear system with symmetric tridiagonal coefficient matrix
    given the cholesky factors. The combination of this function and
    a cholesky factorization to obtain D and L is faster than LAPACK's
    symmetric banded solve (by a factor of ~2 based on brief testing).
    Args:
        D:
        L:
        b:

    Returns:

    """
    y = solve_lin_bidiag(D, L, b, side="L")
    return solve_lin_bidiag(D, L, y, side="U")


def kron_chol_tridiag(D):
    """
    Efficient cholesky decomposition of Kron. prod. of tridiagonals

    Args:
        D: List of N lists, where each each inner list contains the diagonal
        and off diagonal vector of a cholesky factor of a tridiagonal matrix

    Returns:
        LD : The cholesky decomposition of the kronecker product of N tridiag.
        matrices contains 2^N nonzero diagonals. LD contains the non-zero
        diagonal vectors, starting from the main diagonal of the triangular
        factor L, up to the edge of the matrix.

    TODO:
        * Improve documentation
        * There is probably a proper mathematical way to do this
    """
    LD = []
    N = len(D)
    Nd = [len(d[0]) for d in D]
    Nf = np.prod(Nd)

    # Append zero to off-diagonals
    for i, _d in enumerate(D):
        D[i][1] = np.append(D[i][1], 0)

    # Evaluate the list of possible combinations of vector kron. products
    args = [[0, 1]] * N
    for _i, combination in enumerate(product(*args)):
        v = D[-1][combination[-1]]

        for j, c in enumerate(np.flip(combination)[1:]):
            v = np.kron(D[-1 - (j + 1)][c], v)

        # Calculate depth and keep corresponding elements
        coeffs = np.array(combination)
        poly = np.append(np.flip(np.cumprod(Nd[1:])), 1)
        dep = Nf - np.sum(poly * coeffs)
        LD.append(np.array(v[:dep]))

    return LD


def bidiag_mvm(d0, d1, b, side="L"):
    """
    Efficient bidiagonal - vector multiplication
    """
    res = d0 * b

    if side == "L":
        res[1:] += d1 * b[:-1]
    elif side == "U":
        res[:-1] += d1 * b[1:]

    return res


def bidiag_mmm(A, B, side="L"):
    """
    Matrix matrix mutliplication A * B where A is bidiagonal

    Args:
        A: List of lists or list of arrays of bidiagonal coefficients
        B: Numpy array
        side: "L" or "U" for lower and upper side respectively

    Returns:

    """
    # Matrix-matrix multiplication
    res = A[0].reshape(-1, 1) * B

    if side == "L":
        res[1:, :] += A[1].reshape(-1, 1) * B[:-1, :]
    elif side == "U":
        res[:-1, :] += A[1].reshape(-1, 1) * B[1:, :]
    else:
        raise ValueError("Side must be L or U")

    return res


def symtri_mvm(d0, d1, b):
    """
    Efficient symmetric tridiagonal matrix - vector multiplication
    """
    res = d0 * b

    res[1:] += d1 * b[:-1]
    res[:-1] += d1 * b[1:]

    return res


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
    solve_func = lambda A, B: solve_lin_bidiag_mrhs(A[0], A[1], B, side="U")

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

    ND = len(coords)
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
    solve_func = lambda A, B: solve_lin_bidiag_mrhs(A[0], A[1], B, side="U")
    Z = np.zeros((size, Npts))
    for i in prange(size):
        Z[i, :] = kron_op(L, X[i, :], solve_func)

    # Scale by model output
    # TODO: Check that ravel order matches `kron_op`
    if y_model is not None:
        Z = np.ravel(y_model, order="C") * Z

    # Sum the samples
    return Z + S


def get_block_by_index(A, idx_i, idx_j, Nx):
    """
    Get block (idx_i, idx_j) from block matrix A with Nt square blocks of size Nx
    along each dimension
    """
    return A[idx_i * Nx : (idx_i + 1) * Nx, idx_j * Nx : (idx_j + 1) * Nx]


def mult_along_axis(A, B, axis):
    """
    Element-wise multiplication of A and B along the given axis of A

    From: https://stackoverflow.com/questions/
    30031828/multiply-numpy-ndarray-with-1d-array-along-a-given-axis

    Args:
        A:
        B:
        axis:

    Returns:

    """

    A = np.array(A)
    B = np.array(B)

    if axis >= A.ndim:
        raise np.AxisError(axis, A.ndim)
    if A.shape[axis] != B.size:
        raise ValueError(
            "Length of 'A' along the given axis must be the same as B.size"
        )

    shape = np.swapaxes(A, A.ndim - 1, axis).shape
    B_brc = np.broadcast_to(B, shape)
    B_brc = np.swapaxes(B_brc, A.ndim - 1, axis)

    return A * B_brc
