"""
Loglikelihood of timeseries with Exponential correlation and vector noise.
====================
This example shows how to use ``tripy.chol_loglike_1D`` and how it compares to
``scipy.stats.multivariate_normal`` for different numbers of observations
under the assumption of Exponential correlation.

We assume that the measurements are obtained from the following probabilistic model:

.. math::

    \\mathbf{X}_{\\mathrm{model}}(\\mathbf{\\theta}) =
    \\mathbf{K}(\\boldsymbol{\\theta}) \\cdot f(\\mathbf{t})
     + \\mathbf{E}_{\\mathrm{meas}},

where:

* :math:`\\mathbf{X}_{\\mathrm{model}}` is the vector of model predictions.
* :math:`f(\\mathbf{t})` is a vector valued function.
* :math:`\\mathbf{K}(\\boldsymbol{\\theta})` is the correlated multiplicative
 uncertainty factor.
* :math:`\\mathbf{E}_{\\mathrm{meas}}` i.i.d. Gaussian random variables.
* :math:`\\boldsymbol{\\theta}` is a set of parameters of the probabilistic model.

The multiplicative factor :math:`\\mathbf{K}` and additive factor :math:`\\mathbf{E}`
 are distributed as
:math:`\\mathbf{K}(\\boldsymbol{\\theta}) \\sim \\mathcal{N} \left[ 1.0,
\\boldsymbol{\\Sigma}_{\\mathrm{K}}(\\boldsymbol{\\theta})
 \\right]`
and :math:`\\mathbf{E}_{\\mathrm{meas}} \\sim \\mathcal{N}(0,
 \\boldsymbol{\\Sigma}_{\\mathrm{E}}`,
respectively. This yields:

.. math::

        \\mathbf{X}_{\\mathrm{model}}(\\boldsymbol{\\theta}) \\sim \\mathcal{N}
        \\left[ f(\\mathbf{t}), \\mathbf{\\Sigma}(\\boldsymbol{\\theta}) \\right],

with :math:`\\boldsymbol{\\Sigma}(\\boldsymbol{\\theta}) = f(\\mathbf{t}) \\cdot
\\boldsymbol{\\Sigma}_{\\mathrm{K}} + \\boldsymbol{\\Sigma}_{\\mathrm{E}}`
"""

# %%
# Import packages:
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from tripy.kernels import Exponential
from tripy.loglikelihood import chol_loglike_1D

# %%
# Problem setup

N = 100  # Number of points
std_meas = np.repeat(
    2.0, N
)  # Standard deviation of the additive measurement uncertainty
std_model = np.repeat(0.5, N)  # Standard deviation of the modeling uncertainty
l_corr = 10.0  # Correlation lengthscale

# Coordinate vector
t = np.sort(np.linspace(0, 10, N) + 0.1 * np.random.rand(N))

# Vector valued function evaluation
y_func = np.cos(t) + 5

# Observations
v_obs = np.random.rand(N) + y_func

# %%
# Reference solution using scipy:

e_cov_mx = np.diag(std_meas ** 2)
kernel = Exponential(np.reshape(t, (-1, 1)))
corr_mx = kernel.eval(std_model, length_scale=l_corr)
kph_cov_mx = np.matmul(np.diag(y_func), np.matmul(corr_mx, np.diag(y_func)))
cov_mx = kph_cov_mx + e_cov_mx
logL_ref = stats.multivariate_normal.logpdf(v_obs, cov=cov_mx)


# %%
# Timing comparison between conventional and efficient solutions:

N_iter = 10

# Reference
N_vec = np.arange(100, 1000, 100)
t_ref_list = []

for i, N in enumerate(N_vec):

    # Vectors of measurement and modeling uncertainty std. dev.
    std_meas = np.repeat(2.0, N)
    std_model = np.repeat(0.5, N)

    # Coord vector and a test function
    t = np.sort(np.linspace(0, 10, N) + 0.1 * np.random.rand(N))
    y_func = np.cos(t) + 5

    # Observations
    v_obs = np.random.rand(N) + y_func

    e_cov_mx = np.diag(std_meas ** 2)
    kernel = Exponential(np.reshape(t, (-1, 1)))

    # Reference
    t1 = timer()
    for _j in range(N_iter):
        corr_mx = kernel.eval(std_model, length_scale=l_corr)
        kph_cov_mx = np.matmul(np.diag(y_func), np.matmul(corr_mx, np.diag(y_func)))
        cov_mx = kph_cov_mx + e_cov_mx
        test = stats.multivariate_normal.logpdf(v_obs, cov=cov_mx)
    t2 = timer()

    t_ref_list.append((t2 - t1) / N_iter)

    print("=============================")
    print(f"Iter {i + 1} / {len(N_vec)}")
    print(f"N = {N}")
    print(f"t = {t_ref_list[-1]}")
    print("=============================")

# Efficient solution
p_vec = np.linspace(2, 6, 50)
t_list = []
for i, p in enumerate(p_vec):

    N = int(10 ** p)

    # Vectors of measurement and modeling uncertainty std. dev.
    std_meas = np.repeat(2.0, N)
    std_model = np.repeat(0.5, N)

    # Coord vector and a test function
    t = np.sort(np.linspace(0, 10, N) + 0.1 * np.random.rand(N))
    y_func = np.cos(t) + 5

    # Observations
    v_obs = np.random.rand(N) + y_func

    # Initial evaluation to perform jit compilation using numba.
    _ = chol_loglike_1D(v_obs, t, l_corr, std_model, std_meas=std_meas, y_model=y_func)

    # Call function
    t1 = timer()
    for _j in range(N_iter):
        test = chol_loglike_1D(
            v_obs, t, l_corr, std_model, std_meas=std_meas, y_model=y_func
        )
    t2 = timer()

    t_list.append((t2 - t1) / N_iter)

    print("=============================")
    print(f"Iter {i + 1} / {len(p_vec)}")
    print(f"p = {p}")
    print(f"t = {t_list[-1]}")
    print("=============================")

# %%
# Plot a comparison of the wall clock times:

fig = plt.figure()
plt.plot([int(10 ** p) for p in p_vec], t_list, label="Efficient evaluation")
plt.plot(N_vec, t_ref_list, label="Naive evaluation")
plt.xlabel("No. of points")
plt.ylabel("Wall clock time [s]")
plt.xscale("log")
plt.legend(bbox_to_anchor=(0.8, 0.85), bbox_transform=fig.transFigure)
plt.grid()
plt.show()
