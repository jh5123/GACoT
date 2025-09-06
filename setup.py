"""
Module: setup
Purpose: Component of GACoT framework
Author: GACoT Framework
Date: 2025
"""

from setuptools import setup, find_packages

setup(
    name="gacot",
    version="1.0.0",
    author="Jay Huang",
    description="Graph-Augmented Chain of Thought for Financial Reasoning",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "sentence-transformers>=2.2.0", 
        "scikit-learn>=1.0.0", 
        "torch>=1.9.0"
    ],
    extras_require={
        "viz": ["matplotlib>=3.6.0", "seaborn>=0.12.0", "pandas>=2.0.0"],
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "mypy>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "gacot-eval=evaluate:main",
        ],
    },
)