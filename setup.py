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
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'ninja',
        'imageio>=2.28.0',
    ],
)