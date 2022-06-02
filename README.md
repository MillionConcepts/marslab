# marslab

[![DOI](https://zenodo.org/badge/364688103.svg)](https://zenodo.org/badge/latestdoi/364688103)

A library of Python utilities for working with observational data of Mars, 
especially multispectral image data from rovers. This library does not contain 
discrete user-facing applications; if you're looking for a clock, a plotter, or 
an image generator, you've found one of its dependencies.

Feedback is welcomed and encouraged. If the content of your feedback might be 
covered by the MSL or M2020 Team Guidelines, please email: `chase@millionconcepts.com` 
Otherwise, please file an Issue.

## installation

We recommend that you use the ```conda``` package manager to control your Python 
environment for this library. We do not officially support use of non-```conda``` Python
environments (the setup.py file is included for convenience and install hooks only). If 
you're already equipped with ```conda```, please create an env for this package using the following 
commands:

    conda env create -f environment.yml
    conda activate marslab

Otherwise, please follow the rest of the instructions in this file.

### step 1: install conda and support software

*If you already have Anaconda or Miniconda installed on your computer, you can
skip this step. If it's very old and not working well, you should uninstall it first.
We **definitely** don't recommend installing multiple versions of ```conda```
unless you have a really strong need to do so.*

[You can get ```conda``` here as part of the Miniconda distribution of Python](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html).
Download the 64-bit version of the installer for your operating system and
follow the instructions on that website to set up your environment. Make sure
you download Miniconda3, not Miniconda2. ```marslab``` is not compatible with
Python 2.

On Windows, depending on what else you have installed in your environment, you may also need to install
Build Tools for Visual Studio, which you can find on this page: https://visualstudio.microsoft.com/downloads/

### step 2: create conda environment

Now that you have ```conda``` installed, you can set up a Python environment
to use ```marslab```. Open up a terminal: Anaconda Prompt on Windows, Terminal on macOS,
or your console emulator of choice on Linux. Navigate to the directory where
you put the repository. Make sure you have git installed in your base conda environment:

```conda install -n base git```

After that completes, run:

```conda env create -f environment.yml```

Say yes at the prompts and let the installation finish. Then run:

```conda env list```

You should see ```marslab``` in the list of environments. Now run:

```conda activate marslab```

and you will be in a Python environment that contains all the packages
```marslab``` needs. 

This library has an additional dependency on Windows that is not currently accounted for 
in the environment.yml file. Run ```pip install windows-curses``` to install it.

**Important:** now that you've created this environment, you should 
always have it active whenever you work with ```marslab```.

If you can't activate the environment, see 'common gotchas' below.

# common gotchas

* If you get error messages when running a Notebook or other ```marslab``` code, 
  make sure you have activated the```conda``` environment by running ```conda activate marslab```.
* If you use multiple shells on macOS or Linux, ```conda``` will only 
automatically set up the one it detects as your system default. If you can't
activate the environment, check a different shell.
* If you've already got an installed version of ```conda``` on your system, installing
an additional version without uninstalling the old one may make environment setup very
challenging. We do not recommend installing multiple versions of ```conda``` at once
unless you really need to.
