# Changelog

<!--next-version-placeholder-->

## v0.7.3 (2022-02-03)
### Fix
* Added underscore to name of kron_loglike_ND_tridiag ([`3f5442f`](https://gitlab.com/GKoune/tripy/-/commit/3f5442f9d36bb6d2e15b7d7106842efbf6cc7301))

## v0.7.2 (2022-02-03)
### Feature
* Option to add jitter to the covariance matrix when std_meas is zero ([`170d450`](https://gitlab.com/GKoune/tripy/-/commit/170d450c9cdd009bfe85d9f3247a1ce91c11760f))
* Std_meas and y_model are now optional arguments wherever possible. Closes #4 ([`133c7a1`](https://gitlab.com/GKoune/tripy/-/commit/133c7a1a3cf602ad182c9e276f25601499abc6a2))
* Implemented loglikelihood_linear_normal ([`1d006b9`](https://gitlab.com/GKoune/tripy/-/commit/1d006b9aa210b0e5ffe8cd7e4c5f0fc5f3869d8f))
* Fixed loglikelihood.py and updated test_block_cholesky_loglikelihood_2D to account for change in call signature of chol_loglike_2D ([`4f15832`](https://gitlab.com/GKoune/tripy/-/commit/4f15832a8da902cd9b3ad8b349f9e50f7a5bfb10))
* Reworked some call signatures and docstrings for consistency, reworked kron_loglike_2D to work in cases with no tridiagonality. ([`ffa75a8`](https://gitlab.com/GKoune/tripy/-/commit/ffa75a87fd90818ae2c72323875bd748e6fadaeb))
* Utility function for checking and casting standard deviation into arrays ([`227778b`](https://gitlab.com/GKoune/tripy/-/commit/227778b22e61e0e6e88c9fa8d01ed81793c54da6))
* Updated integration test for cholesky 1D likelihood ([`2bdeab1`](https://gitlab.com/GKoune/tripy/-/commit/2bdeab13971589db9ceff2c7c2c603533c80d7a8))
* Reworked chol_loglike_1D to make measurement uncertainty and multiplicative scaling factor optional ([`cd5b377`](https://gitlab.com/GKoune/tripy/-/commit/cd5b377dd07c6cecdb9c8bff6c86c221a897ceb4))
* Added testing for block Cholesky decomposition and loglikelihood ([`6b8c172`](https://gitlab.com/GKoune/tripy/-/commit/6b8c172874446859cdfc0a7e0007f9daee369b4a))
* Added additive 1D cholesky loglikelihood example ([`21b0f9b`](https://gitlab.com/GKoune/tripy/-/commit/21b0f9b353ca0fc5e6c7a7d53bb0ec3ba92c78f3))
* Added pypirc to gitignore ([`d82c786`](https://gitlab.com/GKoune/tripy/-/commit/d82c786b69f14fba2a65749b173eff3bf2913606))
* Setting up semantic versioning ([`b864d1b`](https://gitlab.com/GKoune/tripy/-/commit/b864d1b518b86a744b67c5cb8b1e87c710068ada))
* Changed name in setup.py for PyPI upload ([`2b523c2`](https://gitlab.com/GKoune/tripy/-/commit/2b523c21a8887b5544998e6c355dbee6981300db))
* Removed older/unused code from loglikelihood.py ([`e17d0e9`](https://gitlab.com/GKoune/tripy/-/commit/e17d0e97823b78bdba2521fb0e3228768617eff4))

### Fix
* Disabled kron_loglike_2D_tridiag due to error ([`170e868`](https://gitlab.com/GKoune/tripy/-/commit/170e8683a44f3e4ab12fceeab621328b29bd51a2))
* Fixed error in logic of log_likelihood_linear_normal ([`65419f0`](https://gitlab.com/GKoune/tripy/-/commit/65419f0dd78fead4420f5c9ea146b9bc2abfd490))
* Fixed input shape of chol_loglike_2D ([`36ed7ed`](https://gitlab.com/GKoune/tripy/-/commit/36ed7ed28a46f9ea17f5cc8a2ff2a20d62cced24))
* Added casting of scalars to arrays in inv_cov_vec_1D. Closes #5 ([`699c50e`](https://gitlab.com/GKoune/tripy/-/commit/699c50e37e94299a495479d9aa0316a7d1a26fd5))
* Fixed typo in setup.py ([`d75ead0`](https://gitlab.com/GKoune/tripy/-/commit/d75ead086f6425dde596506c971c2dab7d652d25))
* Setup.py not reading version number when using semantic versioning ([`60ed4a3`](https://gitlab.com/GKoune/tripy/-/commit/60ed4a3b943bd8d15475945fb366540131f4c240))

### Documentation
* Fixed 1D cholesky example ([`b508b7b`](https://gitlab.com/GKoune/tripy/-/commit/b508b7b66edc6bdb45a916f4ee1a314409069847))
* Fixed kron_saple_ND docstring ([`984d63b`](https://gitlab.com/GKoune/tripy/-/commit/984d63bbb9e4bdd53f1da8ed0b4acb12530e3dda))
* Added first test usage example using sphinx-gallery ([`5bc26f9`](https://gitlab.com/GKoune/tripy/-/commit/5bc26f9a571fd68a72dfeceeae579c5c3cde87ef))
* Added docstrings and typehints to sampling.py ([`9d9baf1`](https://gitlab.com/GKoune/tripy/-/commit/9d9baf1080f806345919cc8e9d9e888c9588bf16))
* Improved docstrings in loglikelihood.py and implemented type hinting ([`fd46528`](https://gitlab.com/GKoune/tripy/-/commit/fd465284651f97e58a5b27fe5a2b2ecc9796a208))
* Added sphinx.ext.todo extension to support todo's in docs ([`7e72dcb`](https://gitlab.com/GKoune/tripy/-/commit/7e72dcb350e241412b07a87234f0b8f0aedbdf4b))
* Switched to rtd theme ([`a6ac8f9`](https://gitlab.com/GKoune/tripy/-/commit/a6ac8f92db96ef53a66ded9dc7de8749d0a4a2bb))
* Fixed typo in for_controbitors.md ([`c417907`](https://gitlab.com/GKoune/tripy/-/commit/c4179071e76b795b3b5ecc8166f448cceaf5aa98))
* Fixed conf.py path in readthedocs.yaml ([`451b66f`](https://gitlab.com/GKoune/tripy/-/commit/451b66fbc9bba471c1093773ceb796e43ace019d))
* Updated paths in conf.py ([`8c87b11`](https://gitlab.com/GKoune/tripy/-/commit/8c87b113bc5b36a5127eac9c811125488533bf9d))
* Added readthedocs yaml file and specified documentation requirements ([`d957d67`](https://gitlab.com/GKoune/tripy/-/commit/d957d677256e70c1fa8bbffc9d60c715287aafc7))
* Removed some unused dependencies from setup.py ([`cf2a035`](https://gitlab.com/GKoune/tripy/-/commit/cf2a0354800dad04d7b3592218a1b6805d10d72b))
* Improved documentation and  switched to markdown ([`86d74f9`](https://gitlab.com/GKoune/tripy/-/commit/86d74f91bd1e7de431bd77e50370fb45aba207c2))
* Added initial tests for documentation with Sphinx ([`86c4786`](https://gitlab.com/GKoune/tripy/-/commit/86c4786350e7bb3e7700207b6d3a1170db366fa1))

## v0.7.1 (2022-02-03)


## v0.7.0 (2022-02-03)
### Feature
* Option to add jitter to the covariance matrix when std_meas is zero ([`170d450`](https://gitlab.com/GKoune/tripy/-/commit/170d450c9cdd009bfe85d9f3247a1ce91c11760f))

## v0.6.0 (2022-02-03)
### Feature
* Std_meas and y_model are now optional arguments wherever possible. Closes #4 ([`133c7a1`](https://gitlab.com/GKoune/tripy/-/commit/133c7a1a3cf602ad182c9e276f25601499abc6a2))
* Implemented loglikelihood_linear_normal ([`1d006b9`](https://gitlab.com/GKoune/tripy/-/commit/1d006b9aa210b0e5ffe8cd7e4c5f0fc5f3869d8f))

### Fix
* Fixed error in logic of log_likelihood_linear_normal ([`65419f0`](https://gitlab.com/GKoune/tripy/-/commit/65419f0dd78fead4420f5c9ea146b9bc2abfd490))

## v0.5.0 (2022-02-02)
### Feature
* Fixed loglikelihood.py and updated test_block_cholesky_loglikelihood_2D to account for change in call signature of chol_loglike_2D ([`4f15832`](https://gitlab.com/GKoune/tripy/-/commit/4f15832a8da902cd9b3ad8b349f9e50f7a5bfb10))
* Reworked some call signatures and docstrings for consistency, reworked kron_loglike_2D to work in cases with no tridiagonality. ([`ffa75a8`](https://gitlab.com/GKoune/tripy/-/commit/ffa75a87fd90818ae2c72323875bd748e6fadaeb))
* Utility function for checking and casting standard deviation into arrays ([`227778b`](https://gitlab.com/GKoune/tripy/-/commit/227778b22e61e0e6e88c9fa8d01ed81793c54da6))

### Fix
* Fixed input shape of chol_loglike_2D ([`36ed7ed`](https://gitlab.com/GKoune/tripy/-/commit/36ed7ed28a46f9ea17f5cc8a2ff2a20d62cced24))
* Added casting of scalars to arrays in inv_cov_vec_1D. Closes #5 ([`699c50e`](https://gitlab.com/GKoune/tripy/-/commit/699c50e37e94299a495479d9aa0316a7d1a26fd5))

### Documentation
* Fixed 1D cholesky example ([`b508b7b`](https://gitlab.com/GKoune/tripy/-/commit/b508b7b66edc6bdb45a916f4ee1a314409069847))

## v0.4.0 (2022-01-12)
### Feature
* Updated integration test for cholesky 1D likelihood ([`2bdeab1`](https://gitlab.com/GKoune/tripy/-/commit/2bdeab13971589db9ceff2c7c2c603533c80d7a8))
* Reworked chol_loglike_1D to make measurement uncertainty and multiplicative scaling factor optional ([`cd5b377`](https://gitlab.com/GKoune/tripy/-/commit/cd5b377dd07c6cecdb9c8bff6c86c221a897ceb4))

### Documentation
* Fixed kron_saple_ND docstring ([`984d63b`](https://gitlab.com/GKoune/tripy/-/commit/984d63bbb9e4bdd53f1da8ed0b4acb12530e3dda))
* Added first test usage example using sphinx-gallery ([`5bc26f9`](https://gitlab.com/GKoune/tripy/-/commit/5bc26f9a571fd68a72dfeceeae579c5c3cde87ef))

## v0.3.1 (2022-01-10)
### Documentation
* Added docstrings and typehints to sampling.py ([`9d9baf1`](https://gitlab.com/GKoune/tripy/-/commit/9d9baf1080f806345919cc8e9d9e888c9588bf16))

## v0.3.0 (2022-01-10)
### Feature
* Added testing for block Cholesky decomposition and loglikelihood ([`6b8c172`](https://gitlab.com/GKoune/tripy/-/commit/6b8c172874446859cdfc0a7e0007f9daee369b4a))
* Added additive 1D cholesky loglikelihood example ([`21b0f9b`](https://gitlab.com/GKoune/tripy/-/commit/21b0f9b353ca0fc5e6c7a7d53bb0ec3ba92c78f3))

### Documentation
* Improved docstrings in loglikelihood.py and implemented type hinting ([`fd46528`](https://gitlab.com/GKoune/tripy/-/commit/fd465284651f97e58a5b27fe5a2b2ecc9796a208))
* Added sphinx.ext.todo extension to support todo's in docs ([`7e72dcb`](https://gitlab.com/GKoune/tripy/-/commit/7e72dcb350e241412b07a87234f0b8f0aedbdf4b))

## v0.2.2 (2021-11-29)
### Fix
* Fixed typo in setup.py ([`d75ead0`](https://gitlab.com/GKoune/tripy/-/commit/d75ead086f6425dde596506c971c2dab7d652d25))

## v0.2.1 (2021-11-29)


## v0.2.0 (2021-11-29)
### Feature
* Added pypirc to gitignore ([`d82c786`](https://gitlab.com/GKoune/tripy/-/commit/d82c786b69f14fba2a65749b173eff3bf2913606))

### Fix
* Setup.py not reading version number when using semantic versioning ([`60ed4a3`](https://gitlab.com/GKoune/tripy/-/commit/60ed4a3b943bd8d15475945fb366540131f4c240))

## v0.1.0 (2021-11-29)
### Feature
* Setting up semantic versioning ([`b864d1b`](https://gitlab.com/GKoune/tripy/-/commit/b864d1b518b86a744b67c5cb8b1e87c710068ada))
* Changed name in setup.py for PyPI upload ([`2b523c2`](https://gitlab.com/GKoune/tripy/-/commit/2b523c21a8887b5544998e6c355dbee6981300db))
* Removed older/unused code from loglikelihood.py ([`e17d0e9`](https://gitlab.com/GKoune/tripy/-/commit/e17d0e97823b78bdba2521fb0e3228768617eff4))

### Documentation
* Switched to rtd theme ([`a6ac8f9`](https://gitlab.com/GKoune/tripy/-/commit/a6ac8f92db96ef53a66ded9dc7de8749d0a4a2bb))
* Fixed typo in for_controbitors.md ([`c417907`](https://gitlab.com/GKoune/tripy/-/commit/c4179071e76b795b3b5ecc8166f448cceaf5aa98))
* Fixed conf.py path in readthedocs.yaml ([`451b66f`](https://gitlab.com/GKoune/tripy/-/commit/451b66fbc9bba471c1093773ceb796e43ace019d))
* Updated paths in conf.py ([`8c87b11`](https://gitlab.com/GKoune/tripy/-/commit/8c87b113bc5b36a5127eac9c811125488533bf9d))
* Added readthedocs yaml file and specified documentation requirements ([`d957d67`](https://gitlab.com/GKoune/tripy/-/commit/d957d677256e70c1fa8bbffc9d60c715287aafc7))
* Removed some unused dependencies from setup.py ([`cf2a035`](https://gitlab.com/GKoune/tripy/-/commit/cf2a0354800dad04d7b3592218a1b6805d10d72b))
* Improved documentation and  switched to markdown ([`86d74f9`](https://gitlab.com/GKoune/tripy/-/commit/86d74f91bd1e7de431bd77e50370fb45aba207c2))
* Added initial tests for documentation with Sphinx ([`86c4786`](https://gitlab.com/GKoune/tripy/-/commit/86c4786350e7bb3e7700207b6d3a1170db366fa1))
