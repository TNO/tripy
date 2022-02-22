from typing import Callable, Optional

import numpy as np
import torch
from torch.distributions import MultivariateNormal


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
