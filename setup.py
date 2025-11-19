#this is the setup file for the project
from setuptools import setup, find_packages
setup(
    name="covariance_estimation",
    version="0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
)

