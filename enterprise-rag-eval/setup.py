from setuptools import setup, find_packages

setup(
    name="enterprise-rag-eval",
    version="0.1.0",
    description="Case-aware evaluation framework for enterprise RAG systems using LLM-as-a-judge methodology.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
        "pydantic>=2.0.0",
        "openai>=1.0.0",
    ],
)
