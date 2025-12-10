#!/usr/bin/env python3

from src.agentic_security.fix_cycle import FixCycle

def main():
    # Initialize FixCycle with the vulnerable file
    fixer = FixCycle(
        initial_prompt="""Fix the SQL injection vulnerabilities in this file by:
1. Using parameterized queries
2. Adding input validation
3. Implementing proper error handling
4. Preventing table name injection""",
        files=["tests/samples/vulnerable_query.py"]
    )
    
    # Run the fix cycle
    success = fixer.run_fix_cycle()
    
    if success:
        print("\nSuccessfully fixed SQL injection vulnerabilities!")
        print("Check the generated tests to verify the fixes.")
    else:
        print("\nFailed to fix all vulnerabilities.")
        print("Review the test output for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
