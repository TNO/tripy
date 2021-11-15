"""
This module contains kernel function definitions.

The aim is to have a simple module for specifying correlation functions that allows
for controlling the correlation parameters during inference. Defining each kernel as
a class also makes it possible to automatically check the type of correlation and
choose more efficient likelihood evaluation methods when possible.


NOTE:
    This implementation is based on the kernel module of sklearn (heavily modified),
    which is distributed under a BSD 3 clause license. It seems that this allows for
    modifying and redistributing the source code as long as the original copyright
    is included:

    BSD 3-Clause License

    Copyright (c) 2007-2021 The scikit-learn developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import math

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma, kv


class kernel:
    """
    Kernel definitions and utility functions for performing Bayesian inference.
    Different kernel types are defined with the aim of minimizing the code needed
    to implement them in Bayesian inference. It should be as simple as:

        1. func_covariance = kernel(coordinates)
        2. covariance_matrix = func_covariance.forward(parameters)

    Input:
      coords: Must be a [npts x ndim] array of coordinates

    NOTES:
        * Kernels must be initialized with model uncertainty standard deviation
        and parameters.
        * The `kernels` class is abstract and not to be used directly.

    TODO:
        * Improve documentation
        * Harmonize this class with MeasurementSpaceTimePoints. Right now both classes
         can compile and store the correlation and covariance matrices which can lead
         to misuse.
         * Improve handling of parameters. Remove the `self.param_name` variables
         and only keep a dict of parameter names and values
    """

    def __init__(self, coords: np.ndarray, std_diag: np.ndarray, **params):

        self.coords = coords
        self.std_diag = std_diag
        self.corr_mx = None
        self.cov_mx = None

        # Get input coord shape
        try:
            self.ndim = np.shape(self.coords)[1]
            self.npts = np.shape(self.coords)[0]
        except IndexError:
            raise IndexError(
                f"Coordinates must have shape (Npts, Ndims) but have "
                f"shape {np.shape(self.coords)}"
            )

        # Error if too many dimensions
        if self.ndim > 4:
            raise ValueError(f"Dimension can't be > 4 but is {self.ndim}")

        # Check for one dimensional inputs and if found, put them in a separate vector
        # TODO: Add check to make sure that this vector is ordered.
        # TODO: This is probably not very robust since it compares floats. Fix.
        self._coord_diff = np.diff(self.coords, axis=0)
        self.is_zero_dim = np.any(self._coord_diff, axis=0).__invert__()
        self.vec_x = self.coords[:, self.is_zero_dim]

        # Initialize parameters and set distance to None
        self.dist = None

    def eval(self, std: np.ndarray, **params):
        """
        Evaluate the covariance matrix

        Args:
            std: Diagonal vector of the uncertainty sta. dev.
            **params: Dict. of kernel hyperparameter values

        Returns:

        """
        self.std_diag = np.diag(std)
        return self._evaluate(**params)

    def corr(self, **params):
        if params:
            self.set_params(**params)

        # Calculate distance when evaluation of the kernel is explicitly requested
        # and if it has not already been calculated
        if isinstance(self.dist, type(None)):
            self.dist = squareform(pdist(self.coords, metric="euclidean"))

        return self._corr_mx()


class Independence(kernel):
    def __init__(self, coords, std=None):
        super().__init__(coords, std)
        self.param_list = []

    def set_params(self):
        pass

    def _corr_mx(self):
        return np.diag(np.ones(self.npts))

    def _evaluate(self, **params):
        return np.matmul(np.matmul(self.std_diag, self.corr(**params)), self.std_diag)


class RationalQuadratic(kernel):
    def __init__(self, coords, std=None, length_scale=None, alpha=None):
        super().__init__(coords, std)
        self.param_list = ["length_scale", "alpha"]
        self.length_scale = length_scale
        self.alpha = alpha

    def set_params(self, params):
        self.length_scale = params["length_scale"]
        self.alpha = params["alpha"]

    def _corr_mx(self):
        return (
            1 + self.dist ** 2 / (2 * self.alpha * self.length_scale ** 2)
        ) ** -self.alpha

    def _evaluate(self, **params):
        return np.matmul(np.matmul(self.std_diag, self.corr(**params)), self.std_diag)


class ExpSineSquared(kernel):
    def __init__(self, coords, std=None, length_scale=None, periodicity=None):
        super().__init__(coords, std)
        self.param_list = ["length_scale", "periodicity"]
        self.length_scale = length_scale
        self.periodicity = periodicity

    def set_params(self, **params):
        self.length_scale = params["length_scale"]
        self.periodicity = params["periodicity"]

    def _corr_mx(self):
        return np.exp(
            -2 * (np.sin(np.pi / self.periodicity * self.dist) / self.length_scale) ** 2
        )

    def _evaluate(self, **params):
        return np.matmul(np.matmul(self.std_diag, self.corr(**params)), self.std_diag)


class RBF(kernel):
    def __init__(self, coords, std=None, length_scale=None):
        super().__init__(coords, std)
        self.param_list = ["length_scale"]
        self.length_scale = length_scale

    def set_params(self, **params):
        self.length_scale = params["length_scale"]

    def _corr_mx(self):
        return np.exp(-self.dist ** 2 / (2 * self.length_scale ** 2))

    def _evaluate(self, **params):
        return np.matmul(np.matmul(self.std_diag, self.corr(**params)), self.std_diag)


class Matern(kernel):
    def __init__(self, coords, std=None, length_scale=None, nu=None):
        super().__init__(coords, std)
        self.param_list = ["length_scale", "nu"]
        self.length_scale = length_scale
        self.nu = nu

    def set_params(self, **params):
        self.nu = params["nu"]
        self.length_scale = params["length_scale"]

    def _corr_mx(self):
        dists = self.dist / self.length_scale
        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1.0 + K + K ** 2 / 3.0) * np.exp(-K)
        elif self.nu == np.inf:
            K = np.exp(-(dists ** 2) / 2.0)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = math.sqrt(2 * self.nu) * K
            K.fill((2 ** (1.0 - self.nu)) / gamma(self.nu))
            K *= tmp ** self.nu
            K *= kv(self.nu, tmp)
        return K

    def _evaluate(self, **params):
        return np.matmul(np.matmul(self.std_diag, self.corr(**params)), self.std_diag)


class DampedCosine(kernel):
    def __init__(self, coords, std=None, length_scale=None, wn=None):
        super().__init__(coords, std)
        self.length_scale = length_scale
        self.wn = wn
        self.param_list = ["length_scale", "wn"]

    def set_params(self, **params):
        self.length_scale = params["length_scale"]
        self.wn = params["wn"]

    def _corr_mx(self):
        return np.exp(-self.dist / self.length_scale) * np.cos(self.dist * self.wn)

    def _evaluate(self, **params):
        return np.matmul(np.matmul(self.std_diag, self.corr(**params)), self.std_diag)


class Exponential(kernel):
    def __init__(self, coords, std=None, length_scale=None):
        super().__init__(coords, std)
        self.length_scale = length_scale
        self.param_list = ["length_scale"]

    def set_params(self, **params):
        self.length_scale = params["length_scale"]

    # def inv_cov(self):
    #     return inv_cov_vec_1D(self.vec_x, self.length_scale, self.std_diag)
    #
    # def inv_corr(self):
    #     return inv_cov_vec_1D(self.vec_x, self.length_scale, np.ones(len(self.vec_x)))
    #
    # def inv_chol(self):
    #     return
    #
    # def inv_eig(self):
    #     return

    def _corr_mx(self):
        return np.exp(-self.dist / (self.length_scale))

    def _evaluate(self, **params):
        return np.matmul(np.matmul(self.std_diag, self.corr(**params)), self.std_diag)


class Cauchy(kernel):
    def __init__(self, coords, std=None, length_scale=None, exponent=None):
        super().__init__(coords, std)
        self.param_list = ["length_scale", "exponent"]
        self.length_scale = length_scale
        self.exponent = exponent

    def set_params(self, **params):
        self.length_scale = params["length_scale"]
        self.exponent = params["exponent"]

    def _corr_mx(self):
        return (1 + (self.dist / self.length_scale) ** 2) ** -self.exponent

    def _evaluate(self, **params):
        return np.matmul(np.matmul(self.std_diag, self.corr(**params)), self.std_diag)


class Gaussian(kernel):
    def __init__(self, coords, std=None, length_scale=None):
        super().__init__(coords, std)
        self.param_list = ["length_scale"]
        self.length_scale = length_scale

    def set_params(self, **params):
        self.length_scale = params["length_scale"]

    def _corr_mx(self):
        return np.exp(-((self.dist / self.length_scale) ** 2))

    def _evaluate(self, **params):
        return np.matmul(np.matmul(self.std_diag, self.corr(**params)), self.std_diag)
