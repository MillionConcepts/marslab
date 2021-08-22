from setuptools import setup, find_packages

# this is provided as a convenience for automated install
# hooks. we do not recommend using this file to install
# marslab or its dependencies. please use conda along with
# the provided environment.yml file.

setup(
    name="marslab",
    version="0.9.3",
    url="https://github.com/millionconcepts/marslab.git",
    author="Million Concepts",
    author_email="chase@millionconcepts.com",
    description="Utilities for working with observational data of Mars.",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "pdr",
        "dustgoggles",
        "pillow",
        "jupyter",
        "astropy",
        "fs",
        "scikit-image",
        "sympy",
        "clize",
        "pandas",
        "more-itertools",
        "pathos",
        "cytoolz",
        "rasterio",
        "pathos",
        "hypothesis",
        "pytest",
        "dateutil"
    ],
)
