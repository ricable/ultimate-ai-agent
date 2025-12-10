import unittest
from sparc_cli.calculator import add_numbers, multiply_numbers, divide_numbers

class TestCalculator(unittest.TestCase):
    def test_add_numbers(self):
        self.assertEqual(add_numbers(2, 3), 5)
        self.assertEqual(add_numbers(-1, 1), 0)
        self.assertEqual(add_numbers(0, 0), 0)
        
    def test_multiply_numbers(self):
        self.assertEqual(multiply_numbers(2, 3), 6)
        self.assertEqual(multiply_numbers(-2, 3), -6)
        self.assertEqual(multiply_numbers(0, 5), 0)
        
    def test_divide_numbers(self):
        self.assertEqual(divide_numbers(6, 2), 3)
        self.assertEqual(divide_numbers(-6, 2), -3)
        self.assertEqual(divide_numbers(0, 5), 0)
