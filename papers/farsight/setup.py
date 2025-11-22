"""Setup script for FarSight MLLM package."""
import os
from setuptools import setup, find_packages

setup(
    name="farsight-mllm",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        # Add your dependencies from requirements.txt here if needed
    ],
    author="",
    description="FarSight Multimodal Large Language Model",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
)
