"""
NeuralFlow: A Comprehensive Neural Network Library
"""
from setuptools import setup, find_packages
import os

# Read the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='neuralflow',
    version='1.0.0',
    author='NeuralFlow Team',
    author_email='team@neuralflow.ai',
    description='A flexible and easy-to-use neural network library for rapid prototyping and deployment',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/neuralflow/neuralflow',
    packages=find_packages(exclude=['tests*', 'docs*']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='neural networks deep learning machine learning AI',
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'matplotlib>=3.3.0',
        'tqdm>=4.60.0',
        'scikit-learn>=0.24.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
            'mypy>=0.900',
            'sphinx>=4.0',
            'sphinx-rtd-theme>=0.5',
        ],
        'notebooks': [
            'jupyter>=1.0',
            'jupytext>=1.11',
            'nbconvert>=6.0',
        ],
        'visualization': [
            'seaborn>=0.11',
            'plotly>=5.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'neuralflow-demo=neuralflow.demos.image_classification_demo:main',
        ],
    },
    include_package_data=True,
    package_data={
        'neuralflow': [
            'demos/*.py',
            'notebooks/*.py',
            'utils/*.py',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/neuralflow/neuralflow/issues',
        'Source': 'https://github.com/neuralflow/neuralflow',
        'Documentation': 'https://neuralflow.readthedocs.io',
    },
)