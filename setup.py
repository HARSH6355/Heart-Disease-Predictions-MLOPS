from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """Return install requirements from a requirements file."""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="MLOPS_Project",
    version="0.0.1",
    author="Harsh Trivedi",
    author_email="harsh6355@gmail.com",
    description="MLOps pipeline for UCI Heart Disease classification",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.10",
)
