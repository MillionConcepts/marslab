from setuptools import setup, find_packages

# this is provided as a convenience for automated install
# hooks. we do not recommend using this file to install
# marslab or its dependencies. please use conda along with
# the provided environment.yml file.

setup(
    name="marslab",
    version="0.9.8",
    url="https://github.com/millionconcepts/marslab.git",
    author="Million Concepts",
    author_email="chase@millionconcepts.com",
    description="Utilities for working with observational data of Mars.",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "dustgoggles",
        "fs",
        "clize",
        "pandas",
        "more-itertools",
        "pathos",
        "cytoolz",
    ],
    extras_require={
        "pdr_load": ["pdr"],
        "tests": ["pytest", "hypothesis"],
        "render": ["pillow", "matplotlib"],
        "notebooks": ["jupyter"],
        "regions": ["astropy"],
        "time": ["sympy", "python-dateutil"],
        "geom": ["pdr"]
    }
)
