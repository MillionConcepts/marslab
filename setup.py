from setuptools import setup, find_packages

# this is provided as a convenience for automated install
# hooks. we do not recommend using this file to install
# marslab or its dependencies. please use conda along with
# the provided environment.yml file.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="marslab",
    version="0.9.21",
    url="https://github.com/millionconcepts/marslab.git",
    author="Million Concepts",
    author_email="chase@millionconcepts.com",
    description="Utilities for working with observational data of Mars.",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "cytoolz",
        "dustgoggles",
        "more-itertools",
        "numpy",
        "pandas",
        "pathos",
        "scipy",
    ],
    extras_require={
        "pdr_load": ["pdr"],
        "tests": ["pytest", "hypothesis"],
        "render": ["pillow", "matplotlib"],
        "notebooks": ["jupyter"],
        "regions": ["astropy"],
        "time": ["sympy", "python-dateutil"],
        "geom": ["pdr"],
        "strict_reshape": ["sympy"],
        "histograms": ["fast-histogram"],
        "masking": ["scikit-image", "opencv"]
    }
)
