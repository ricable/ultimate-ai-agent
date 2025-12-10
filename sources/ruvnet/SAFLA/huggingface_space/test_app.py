#!/usr/bin/env python3
"""
Simple test script to verify the SAFLA app works without queue errors.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from app import SAFLAApp

def test_app_creation():
    """Test that the app can be created and interface built."""
    print("Creating SAFLA app...")
    app = SAFLAApp()
    
    print("Building interface...")
    interface = app.create_interface()
    
    print("Testing system status...")
    status = app._get_system_status()
    print(f"System status: {status}")
    
    print("✅ App creation and interface building successful!")
    return True

if __name__ == "__main__":
    try:
        test_app_creation()
        print("\n🎉 All tests passed! The app is ready to run.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)