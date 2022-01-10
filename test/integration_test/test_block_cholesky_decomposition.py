"""
Testing for the ND Cholesky loglikelihood with vector noise and associated functions:

    * symm_tri_block_chol: Block Cholesky decomposition of block symmetric
    tridiagonal matrices
"""

# ============================================================================
# Imports
# ============================================================================
import numpy as np

from tripy.kernels import Exponential
from tripy.utils import get_block_by_index, inv_cov_vec_1D, symm_tri_block_chol


def test_block_cholesky_decomposition():
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
    jitter = 0.0

    # Initialize reference kernels
    kernel_x = Exponential(np.reshape(x, (-1, 1)), np.ones(Nx), length_scale=lcorr_x)
    kernel_t = Exponential(np.reshape(t, (-1, 1)), std_model, length_scale=lcorr_t)

    # Check covariance
    cov_mx_x = kernel_x.eval(np.ones(Nx)) + np.diag(np.ones(Nx) * jitter)
    cov_mx_t = kernel_t.eval(std_model)
    inv_cov_mx_x = np.linalg.inv(cov_mx_x)
    inv_cov_mx_t = np.linalg.inv(cov_mx_t)

    # Get inverse cov vector in space and time
    Cx_0 = np.diag(inv_cov_mx_x)
    Cx_1 = np.diag(inv_cov_mx_x, k=1)
    Ct_0, Ct_1 = inv_cov_vec_1D(t, lcorr_t, std_model)

    # =====================================================================
    # Test block Cholesky decomposition
    #
    # The following cases are checked:
    #   * With/without multiplicative uncertainty
    #   * With/without tridiagonality in space
    # =====================================================================

    # =====================================================================
    # Tridiagonality in both space and time
    # =====================================================================

    # No scaling vector
    # ---------------------------------------------------------------------
    Cx = [Cx_0, Cx_1]
    Ct = [Ct_0, Ct_1]
    y = np.ones(Nx * Nt)
    GWG = y ** 2 * (1 / std_meas ** 2)
    cov_mx_ref = np.kron(inv_cov_mx_t, inv_cov_mx_x) + np.diag(GWG)

    L_test, C_test = symm_tri_block_chol(Cx, Ct, std_meas ** 2, y=y)
    L_ref = np.linalg.cholesky(cov_mx_ref)

    # Check diagonal blocks
    for i in range(Nt):
        blck_L_ref = get_block_by_index(L_ref, i, i, Nx)
        assert np.allclose(blck_L_ref, L_test[i])

    # Check off diagonal blocks
    for i in range(Nt - 1):
        blck_C_ref = get_block_by_index(L_ref, i + 1, i, Nx)
        assert np.allclose(blck_C_ref, C_test[i])

    # With scaling vector
    # ---------------------------------------------------------------------
    Cx = [Cx_0, Cx_1]
    Ct = [Ct_0, Ct_1]
    y = np.random.rand(Nx * Nt) * 0.1
    GWG = y ** 2 * (1 / std_meas ** 2)
    cov_mx_ref = np.kron(inv_cov_mx_t, inv_cov_mx_x) + np.diag(GWG)

    # Without scaling vector
    L_test, C_test = symm_tri_block_chol(Cx, Ct, std_meas ** 2, y=y)
    L_ref = np.linalg.cholesky(cov_mx_ref)

    # Check diagonal blocks
    for i in range(Nt):
        blck_L_ref = get_block_by_index(L_ref, i, i, Nx)
        assert np.allclose(blck_L_ref, L_test[i])

    # Check off diagonal blocks
    for i in range(Nt - 1):
        blck_C_ref = get_block_by_index(L_ref, i + 1, i, Nx)
        assert np.allclose(blck_C_ref, C_test[i])

    # =====================================================================
    # Tridiagonality in time only
    # =====================================================================

    # No scaling vector
    # ---------------------------------------------------------------------
    Cx = inv_cov_mx_x
    Ct = [Ct_0, Ct_1]
    y = np.ones(Nx * Nt)
    GWG = y ** 2 * (1 / std_meas ** 2)
    cov_mx_ref = np.kron(inv_cov_mx_t, inv_cov_mx_x) + np.diag(GWG)

    # Without scaling vector
    L_test, C_test = symm_tri_block_chol(Cx, Ct, std_meas ** 2, y=y)
    L_ref = np.linalg.cholesky(cov_mx_ref)

    # Check diagonal blocks
    for i in range(Nt):
        blck_L_ref = get_block_by_index(L_ref, i, i, Nx)
        assert np.allclose(blck_L_ref, L_test[i])

    # Check off diagonal blocks
    for i in range(Nt - 1):
        blck_C_ref = get_block_by_index(L_ref, i + 1, i, Nx)
        assert np.allclose(blck_C_ref, C_test[i])

    # With scaling vector
    # ---------------------------------------------------------------------
    Cx = inv_cov_mx_x
    Ct = [Ct_0, Ct_1]
    y = np.random.rand(Nx * Nt) * 0.1
    GWG = y ** 2 * (1 / std_meas ** 2)
    cov_mx_ref = np.kron(inv_cov_mx_t, inv_cov_mx_x) + np.diag(GWG)

    # Without scaling vector
    L_test, C_test = symm_tri_block_chol(Cx, Ct, std_meas ** 2, y=y)
    L_ref = np.linalg.cholesky(cov_mx_ref)

    # Check diagonal blocks
    for i in range(Nt):
        blck_L_ref = get_block_by_index(L_ref, i, i, Nx)
        assert np.allclose(blck_L_ref, L_test[i])

    # Check off diagonal blocks
    for i in range(Nt - 1):
        blck_C_ref = get_block_by_index(L_ref, i + 1, i, Nx)
        assert np.allclose(blck_C_ref, C_test[i])
