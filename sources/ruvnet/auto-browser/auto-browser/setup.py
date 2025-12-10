from setuptools import setup, find_packages

setup(
    name="auto-browser",
    version="0.1.0",
    description="Browser automation CLI tool for configurable site scraping",
    author="rUv",
    author_email="ruv@ruv.net",
    url="https://github.com/yourusername/auto-browser",
    packages=find_packages(),
    scripts=['auto-browser'],
    install_requires=[
        "click>=8.1.0,<9.0.0",
        "pydantic>=2.0.0,<3.0.0",
        "pyyaml>=6.0.1,<7.0.0",
        "rich>=13.0.0,<14.0.0",
        "browser-use>=0.1.23",
        "langchain-openai>=0.2.14",
        "python-dotenv>=1.0.0"
    ],
    python_requires=">=3.11",
)
