"""
UAP CLI Setup Configuration

Unified Agentic Platform Command Line Interface
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "UAP CLI - Command-line tools for the Unified Agentic Platform"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "uap-sdk>=1.0.0",
        "click>=8.0.0",
        "rich>=13.0.0",
        "pyyaml>=6.0.0",
        "psutil>=5.9.0",
        "httpx>=0.24.0"
    ]

setup(
    name="uap-cli",
    version="1.0.0",
    author="UAP Development Team",
    author_email="dev@uap.ai",
    description="UAP CLI - Command-line tools for the Unified Agentic Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uap/uap-cli",
    project_urls={
        "Documentation": "https://docs.uap.ai/cli",
        "Source": "https://github.com/uap/uap-cli",
        "Bug Reports": "https://github.com/uap/uap-cli/issues",
        "Changelog": "https://github.com/uap/uap-cli/blob/main/CHANGELOG.md"
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Framework :: AsyncIO",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0"
        ],
        "full": [
            "uvloop>=0.17.0",  # Performance improvement for asyncio
            "orjson>=3.8.0",   # Faster JSON serialization
            "docker>=6.0.0",   # Docker integration
            "kubernetes>=26.0.0"  # Kubernetes integration
        ]
    },
    entry_points={
        "console_scripts": [
            "uap=uap_cli.main:sync_main",
        ],
    },
    include_package_data=True,
    package_data={
        "uap_cli": [
            "templates/*.py",
            "templates/*.json",
            "templates/*.yaml"
        ]
    },
    keywords=[
        "ai", "agents", "cli", "command-line", "tools", "deployment",
        "management", "automation", "workflow", "devops"
    ],
    zip_safe=False,
    license="MIT",
    platforms=["any"]
)