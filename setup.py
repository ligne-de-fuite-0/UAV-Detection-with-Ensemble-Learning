from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="uav-detection",
    version="1.0.0",
    author="ligne-de-fuite-0",
    description="UAV Detection with Ensemble Learning using ResNet-20 and ConvNeXt-Tiny",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ligne-de-fuite-0/uav-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="uav detection computer-vision pytorch ensemble-learning",
    project_urls={
        "Bug Reports": "https://github.com/ligne-de-fuite-0/uav-detection/issues",
        "Source": "https://github.com/ligne-de-fuite-0/uav-detection",
    },
)