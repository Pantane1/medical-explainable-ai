from setuptools import setup, find_packages

setup(
    name="medical-explainable-ai",
    version="2.1.0",
    description="Explainable AI system for clinical decision support",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "shap>=0.42",
        "lime>=0.2",
        "flask>=2.3",
        "plotly>=5.15",
    ],
)
