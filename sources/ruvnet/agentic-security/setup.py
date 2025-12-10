from setuptools import setup, find_packages

setup(
    name="agentic-security",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'pytest==7.4.3',
        'requests==2.30.0',
        'python-dotenv==1.0.0',
        'pyyaml==6.0.1',
        'openai>=0.27.6',
        'anthropic>=0.5.0',
        'click==8.1.7',
    ],
    entry_points={
        'console_scripts': [
            'agentic-security=agentic_security.security_cli:cli',
        ],
    },
    python_requires='>=3.10',
    author="rUv",
    author_email="your.email@example.com",
    description="AI-powered security scanning and fixing pipeline",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://agentic-security.io",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
