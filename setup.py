# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fpdet",
    version="1.0.1",
    author="Pavel Gostev",
    author_email="gostev.pavel@physics.msu.ru",
    description=" Basic analytical functions for a few-photon light detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vongostev/fpdet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
    ],
)
