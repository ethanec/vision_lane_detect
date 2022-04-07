from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    #version="1.10.0",
    packages=['utils'],
    package_dir={'': 'script/self_localization'})

setup(**setup_args)