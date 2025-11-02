from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name = "MultiAIAgent",
    version="0.1",
    author="NamanChhaparia",
    packages=find_packages(),
    install_requires = requirements
)