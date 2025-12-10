import unittest
import sqlite3
import os
from tests.samples.vulnerable_query import get_user_data, search_users

class TestSQLInjectionFixes(unittest.TestCase):
    def setUp(self):
        # Create test database
        self.db_name = 'test_users.db'
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        
        # Create test table
        self.cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT
            )
        ''')
        
        # Insert test data
        self.cursor.executemany(
            'INSERT INTO users (name, email) VALUES (?, ?)',
            [
                ('Alice', 'alice@test.com'),
                ('Bob', 'bob@test.com'),
                ('Charlie', 'charlie@test.com')
            ]
        )
        self.conn.commit()

    def tearDown(self):
        # Clean up test database
        self.conn.close()
        if os.path.exists(self.db_name):
            os.remove(self.db_name)

    def test_get_user_data_safe(self):
        """Test that get_user_data is safe from SQL injection"""
        # Test normal case
        result = get_user_data(1, 'users')
        self.assertEqual(len(result), 1)
        
        # Test SQL injection attempt
        result = get_user_data("1 OR 1=1", 'users')
        self.assertEqual(len(result), 0)  # Should fail safely
        
        # Test table name injection
        result = get_user_data(1, 'users; DROP TABLE users;')
        self.assertEqual(len(result), 0)  # Should fail safely

    def test_search_users_safe(self):
        """Test that search_users is safe from SQL injection"""
        # Test normal case
        result = search_users('Alice')
        self.assertEqual(len(result), 1)
        
        # Test SQL injection attempt
        result = search_users("' OR '1'='1")
        self.assertEqual(len(result), 0)  # Should fail safely
        
        # Test UNION injection attempt
        result = search_users("' UNION SELECT * FROM users--")
        self.assertEqual(len(result), 0)  # Should fail safely

if __name__ == '__main__':
    unittest.main()
