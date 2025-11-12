from setuptools import setup, find_packages

setup(
    name="MetaIR-Framework",
    version="0.1.0",
    author="Juscimara Avelino",
    description="A meta-learning framework for imbalanced regression recommendation",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "seaborn",
        "matplotlib",
        "ImbalancedLearningRegression",
        "smogn",
        "resreg"
    ],
    python_requires=">=3.8",
)
