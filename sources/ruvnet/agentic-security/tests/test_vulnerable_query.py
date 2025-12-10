import unittest
import os
import sqlite3
from tests.samples.vulnerable_query import *

class TestVulnerable_Query(unittest.TestCase):
    def setUp(self):
        """Set up test database"""
        self.db_name = "test_users.db"
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        
        # Create test table
        self.cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT
            )
        """)
        
        # Insert test data
        self.cursor.executemany(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            [
                ("Alice", "alice@test.com"),
                ("Bob", "bob@test.com")
            ]
        )
        self.conn.commit()

    def tearDown(self):
        """Clean up test database"""
        self.conn.close()
        if os.path.exists(self.db_name):
            os.remove(self.db_name)

    def test_valid_input(self):
        """Test with valid input"""
        result = get_user_data("1", "users")
        self.assertIsNotNone(result)
        self.assertEqual(result[0][1], "Alice")

    def test_sql_injection_prevention(self):
        """Test SQL injection prevention"""
        malicious_inputs = [
            "1 OR 1=1",
            "1; DROP TABLE users;",
            "1' UNION SELECT * FROM users--",
            "1) OR '1'='1",
            "1/**/OR/**/1=1"
        ]
        for input_val in malicious_inputs:
            with self.assertRaises(Exception):
                get_user_data(input_val, "users")

    def test_table_name_validation(self):
        """Test table name validation"""
        invalid_tables = [
            "users;",
            "users--",
            "users DROP TABLE users",
            "non_existent_table",
            "users/**/WHERE/**/1=1"
        ]
        for table in invalid_tables:
            with self.assertRaises(Exception):
                get_user_data("1", table)

    def test_search_users_validation(self):
        """Test search_users function validation"""
        # Valid search
        result = search_users("Alice")
        self.assertIsNotNone(result)
        self.assertEqual(result[0][1], "Alice")
        
        # SQL injection attempts
        malicious_searches = [
            "Alice' OR '1'='1",
            "Alice; DROP TABLE users;",
            "Alice' UNION SELECT * FROM users--",
            "%' OR name LIKE '%",
            "Alice/**/OR/**/1=1"
        ]
        for search in malicious_searches:
            with self.assertRaises(Exception):
                search_users(search)

    def test_error_handling(self):
        """Test error handling for edge cases"""
        # Test with None values
        with self.assertRaises(Exception):
            get_user_data(None, "users")
        with self.assertRaises(Exception):
            get_user_data("1", None)
        with self.assertRaises(Exception):
            search_users(None)
        
        # Test with empty strings
        with self.assertRaises(Exception):
            get_user_data("", "users")
        with self.assertRaises(Exception):
            get_user_data("1", "")
        with self.assertRaises(Exception):
            search_users("")

if __name__ == "__main__":
    unittest.main()
