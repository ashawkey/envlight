import os
from setuptools import setup

setup(
    name='envlight', # package name, import this to use python API
    version='0.1.0',
    description='environment lighting toolkit',
    url='https://github.com/ashawkey/envlight',
    author='kiui',
    author_email='ashawkey1999@gmail.com',
    packages=['envlight'],
    install_requires=[
        'torch',
        'numpy',
        'ninja',
        'imageio',
        'nvdiffrast',
    ],
)