import unittest
import os
import tempfile
from src.agentic_security.fix_cycle import FixCycle

class TestFixCycle(unittest.TestCase):
    def setUp(self):
        # Create a temporary test file
        self.test_file = tempfile.NamedTemporaryFile(suffix='.py', delete=False)
        self.test_file.write(b"""
def add(a, b):
    return a + b  # Basic function for testing
""")
        self.test_file.close()
        
        # Create test directory if it doesn't exist
        os.makedirs("tests", exist_ok=True)
        
        self.fix_cycle = FixCycle(
            initial_prompt="Fix the add function to handle string inputs",
            files=[self.test_file.name]
        )

    def tearDown(self):
        # Clean up temporary file
        if os.path.exists(self.test_file.name):
            os.unlink(self.test_file.name)

    def test_generate_test(self):
        test_file = self.fix_cycle.generate_test(self.test_file.name)
        self.assertTrue(os.path.exists(test_file))
        
    def test_validate_test(self):
        # Create a valid test file
        test_content = """
import unittest

class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)

if __name__ == '__main__':
    unittest.main()
"""
        test_file = os.path.join("tests", "test_valid.py")
        with open(test_file, 'w') as f:
            f.write(test_content)
            
        self.assertTrue(self.fix_cycle.validate_test(test_file))
        
        # Clean up
        os.unlink(test_file)

    def test_run_fix_cycle(self):
        # This is a basic test - in practice, would need more complex scenarios
        result = self.fix_cycle.run_fix_cycle()
        self.assertIsInstance(result, bool)

if __name__ == '__main__':
    unittest.main()
