import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="linear-imputation",
    version="1.0.0",
    description="Fill-in missing values using data mean and correlation",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/thinknum/linear-data-imputation",
    author="Thinknum",
    author_email="matias.morant@thinknum.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    packages=["linear_imputation"],
    include_package_data=True,
    install_requires=["numpy", "pandas", "scipy"]
)