from setuptools import setup, find_packages

setup(
    name="finrl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "stable-baselines3",
        "alpaca-trade-api>=3.0.0",
        "gymnasium>=0.28.1",
        "torch>=1.13.1",
        # other dependencies from requirements.txt
    ]
) 