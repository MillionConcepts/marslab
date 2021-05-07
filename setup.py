from setuptools import setup, find_packages

setup(
    name='marslab',
    version='0.2.0',
    url='https://github.com/millionconcepts/marslab.git',
    author='Million Concepts',
    author_email='chase@millionconcepts.com',
    description='Utilities for working with observational data of Mars.',
    packages=find_packages(),    
    install_requires=[
    'python >= 3.9.2',
    'numpy >= 1.20.2', 
    'matplotlib >= 3.4.1',
	'pdr >= 0.4.2a0',
 	'opencv >= 4.5.2',
 	'pillow >= 8.1.2',
 	'jupyter >= 1.0.0',
 	'astropy >= 4.2.1',
 	'fs >= 2.4.11',
 	'scikit-image >= 0.18.1',
 	'sympy >= 1.8',
 	'clize >= 4.1.1'
    ]
)
