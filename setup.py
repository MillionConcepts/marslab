from setuptools import setup, find_packages

setup(
    name='marslab',
    version='0.2.0',
    url='https://github.com/millionconcepts/marslab.git',
    author='Million Concepts',
    author_email='chase@millionconcepts.com',
    description='Utilities for working with observational data of Mars.',
    packages=find_packages(),    
    python_requires='>=3.9',
    install_requires=[
	'numpy', 
	'matplotlib',
        'scipy',
	'pdr',
	'opencv',
	'pillow',
	'jupyter',
	'astropy',
	'fs',
	'scikit-image',
	'sympy',
	'clize'
    ]
)
