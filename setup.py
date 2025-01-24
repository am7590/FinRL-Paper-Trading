from setuptools import setup, find_packages

setup(
    name="finrl-paper-trading",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.68.0",
        "uvicorn[standard]==0.15.0",
        "websockets==10.0",
        "autogen==1.0.0",
        "python-dotenv==0.19.0",
        "pytest==7.4.4",
        "pytest-asyncio==0.23.5",
        "finrl==0.3.5",
        "stable-baselines3==2.0.0",
    ],
) 