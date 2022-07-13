# tripy

[![Documentation Status](https://readthedocs.org/projects/lightkde/badge/?version=stable)](https://lightkde.readthedocs.io/en/stable/)
[![CI](https://github.com/TNO/tripy/actions/workflows/push.yml/badge.svg)](https://github.com/TNO/tripy/actions)
[![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/JanKoune/08985bf02bfbea845659e2a487ba86d5/raw/tripy_master_coverage.json)](https://en.wikipedia.org/wiki/Code_coverage)
[![PyPI version](https://img.shields.io/pypi/v/tri-py)](https://pypi.org/project/tri-py/)
[![Conda version](https://anaconda.org/GKoune/tri-py/badges/version.svg)](https://anaconda.org/GKoune/tri-py)
![python versions](https://img.shields.io/pypi/pyversions/tri-py)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A package for efficient[^1] likelihood evaluation and sampling for Multivariate Normal distributions where the covariance matrix:

* Is Separable, i.e. can be expressed as the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) of the covariance over different dimensions (e.g. space and time);
* May have Exponential correlation (i.e. [(block-) tridiagonal](https://en.wikipedia.org/wiki/Tridiagonal_matrix) precision matrix) in one or more dimensions;
* Is polluted with uncorrelated scalar or vector noise.

[^1]: In the general case, exact likelihood evaluation has *O(N<sup>3</sup>)* computational complexity and *O(N<sup>2</sup>)* memory requirements. The term "efficient" is used here to refer to the reduction of complexity and memory usage by utilizing the sparsity and Kronecker product structure of the covariance matrix.

## Structure
**base**: Base class for problem formulation, taken from [taralli](https://gitlab.com/tno-bim/taralli). Likely to be removed in a future update.

**utils**: Utility functions for efficient linear algebra invovling tridiagonal and Kronecker product matrices.

**loglikelihood**: Functions for efficient loglikelihood evaluation.

**kernels**: Formulation of commonly used kernels.

**sampling**: Functions for efficient sampling.

## Usage
![Usage flowchart](docs/figures/loglikelihood_selection_flowchart.png "Loglikelihood function selection flowchart")

## TODOs
* Validation of all functions against reference implementations.
* Documentation, including examples and timing tests.
* Unit and integration testing.
* Improve this README by including mathematical notation and references.
