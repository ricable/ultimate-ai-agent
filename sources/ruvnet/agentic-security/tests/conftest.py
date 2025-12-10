import pytest
import os
from dotenv import load_dotenv

def pytest_configure(config):
    """Load environment variables before running tests"""
    load_dotenv(verbose=True)  # Load environment variables from .env file
