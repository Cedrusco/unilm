#!/usr/bin/env python3
import torch
from setuptools import find_packages, setup

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 4], "Requires PyTorch >= 1.4"

setup(
    name="layoutlm",
    version="0.0",
    author="Yiheng Xu",
    url="https://github.com/microsoft/unilm/tree/master/layoutlm",
    description="LayoutLM",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.6",
    install_requires=[
        "transformers==2.9.0",
        "tensorboardX==2.0",
        "lxml==4.5.1",
        "seqeval==0.0.12",
        "Pillow==7.1.2",
        "Flask==1.1.2",
        "waitress==1.4.4",
        "flask_cors==3.0.8",
        "pandas==1.0.5"
    ],
    extras_require={
        "dev": ["flake8==3.8.2", "isort==4.3.21", "black==19.10b0", "pre-commit==2.4.0"]
    },
)
