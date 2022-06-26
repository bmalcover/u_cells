""" Configuration file for pypi package."""
from os import path

from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as fp:
    install_requires = fp.read()

setup(
    name="u_rpn",
    version="0.10",
    description="Combinations of the U-Net and RPN",
    url="https://github.com/bmalcover/u_cells",
    author="Miquel Miró Nicolau, Dr. Gabriel Moyà Alcover",
    author_email="miquelca32@gmail.com, gabriel_moya@uib.es",
    license="MIT",
    packages=["u_rpn"],
    keywords=["Instance segmentation", "Deep Learning", "Computer Vision"],
    install_requires=install_requires,
    python_requires=">=3.8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
)
