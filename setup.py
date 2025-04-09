from setuptools import setup, find_packages

setup(
    name="my_ml_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "nltk",
        "torch",
        "tensorflow",
        "gensim",
        "scikit-learn",
        "openpyxl",
    ],
    entry_points={
        "console_scripts": [
            "run_training=src.training:main",
        ],
    },
)
