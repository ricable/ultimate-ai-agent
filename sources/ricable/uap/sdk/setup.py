"""
UAP SDK Setup Configuration

Unified Agentic Platform Software Development Kit
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "UAP SDK - Build powerful AI agents and integrations"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "httpx>=0.24.0",
        "websockets>=11.0.0",
        "pyyaml>=6.0.0",
        "aiofiles>=23.0.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.7.0"
    ]

setup(
    name="uap-sdk",
    version="1.0.0",
    author="UAP Development Team",
    author_email="dev@uap.ai",
    description="UAP SDK - Build powerful AI agents and integrations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uap/uap-sdk",
    project_urls={
        "Documentation": "https://docs.uap.ai/sdk",
        "Source": "https://github.com/uap/uap-sdk",
        "Bug Reports": "https://github.com/uap/uap-sdk/issues",
        "Changelog": "https://github.com/uap/uap-sdk/blob/main/CHANGELOG.md"
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
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
        "cli": [
            "uap-cli>=1.0.0"
        ],
        "full": [
            "uap-cli>=1.0.0",
            "uvloop>=0.17.0",  # Performance improvement for asyncio
            "orjson>=3.8.0",   # Faster JSON serialization
            "psutil>=5.9.0"    # System monitoring
        ]
    },
    # Entry points would be added here for CLI tools
    # entry_points={
    #     "console_scripts": [
    #         "uap-sdk=uap_sdk.cli:main",
    #     ],
    # },
    include_package_data=True,
    package_data={
        "uap_sdk": [
            "templates/*.py",
            "templates/*.json",
            "examples/*.py"
        ]
    },
    keywords=[
        "ai", "agents", "artificial-intelligence", "sdk", "framework",
        "chat", "websocket", "api", "automation", "workflow", "plugin"
    ],
    zip_safe=False,
    license="MIT",
    platforms=["any"]
)