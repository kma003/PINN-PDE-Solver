import os
from setuptools import setup, find_packages
from setuptools.command.install import install

requirements = [
	"numpy",
	"matplotlib",
]

setup(
	name="PINN-PDE-Solver",
	version="0.1",
	packages=find_packages(),
	install_requires=requirements,
	test_suite="tests"
)
