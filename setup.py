from setuptools import setup, find_packages

setup(
    name="hallucination-detection-llm",
    version="0.1.0",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "transformers>=4.40",
        "datasets",
        "sentence-transformers",
        "scikit-learn",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "tqdm",
        "pyyaml",
    ],
)
