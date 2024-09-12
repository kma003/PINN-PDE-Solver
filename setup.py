import os
from setuptools import setup, find_packages
from setuptools.command.install import install

dirs = [
    "data",
    "networks/saved_models"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

requirements = [
	"numpy<2.0.0",
	"matplotlib",
	"scipy",
	"torch",
]

setup(
	name="PINN-PDE-Solver",
	version="0.1",
	packages=find_packages(),
	install_requires=requirements,
	test_suite="tests"
)
