"""
Database package for RAG AI agent.
"""
from database.setup import SupabaseClient, setup_database_tables

__all__ = [
    'SupabaseClient',
    'setup_database_tables'
]
