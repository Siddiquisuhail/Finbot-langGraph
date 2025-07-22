# setup.py
from setuptools import setup, find_packages

setup(
    name="ai-financial-analysis",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "langchain>=0.1.0", 
        "langchain-openai>=0.0.5",
        "langgraph>=0.0.20",
        "langsmith>=0.0.70",
        "yfinance>=0.2.18",
        "pandas>=2.0.0",
        "numpy>=1.24.0", 
        "plotly>=5.15.0",
        "requests>=2.31.0",
        "python-dateutil>=2.8.2"
    ],
    python_requires=">=3.8",
    author="AI Financial Analysis System",
    description="Advanced AI-powered financial analysis using LangChain ecosystem",
)
