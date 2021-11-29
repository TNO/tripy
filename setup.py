import os

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Package requirements
user_requires = [
    "numpy<2",
    "scipy<2",
    "nestle<1",
    "emcee<4",
    "dynesty<2",
    "matplotlib<4",
    "tabulate<1",
    "tqdm<5",
    "numba<1",
    "torch<2",
]
doc_requires = [
    "six",
    "sphinx<5",
    "sphinx-copybutton<1",
    "sphinx-inline-tabs",
    "sphinxcontrib-bibtex<3",
    "myst-parser<1",
    "sphinx-rtd-theme<2",
    "sphinx-rtd-dark-model<2",
]
test_requires = [
    "pytest",
    "coverage",
]
packaging_requires = [
    "twine",
]
format_requires = [
    "pre-commit",
    "black",
    "flake8",
    "flake8-tidy-imports",
    "flake8-import-order",
]

# In order to reduce the time to build the documentation
# https://github.com/readthedocs/readthedocs.org/issues/5512#issuecomment-475073310
on_ci_doc = os.environ.get("CI_DOC") is not None
if on_ci_doc:
    install_requires = doc_requires
else:
    install_requires = user_requires

setuptools.setup(
    name="tri-py",
    version="0.0.2",
    author="Ioannis Koune",
    author_email="G.Koune@gmail.com",
    description="A package for efficient loglikelihood evaluation with"
    " structured covariance matrices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/GKoune/tripy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "tripy"},
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    # additional packages for development (additional to `install_requires`)
    extras_require={
        "development": [
            *doc_requires,
            *test_requires,
            *packaging_requires,
            *format_requires,
        ],
        "testing": test_requires,
        "docs": doc_requires,
    },
    python_requires=">=3.6",
)
