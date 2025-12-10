"""
Test script to launch the SAFLA app and verify it works.
"""

import sys
import time
import threading
from app import SAFLAApp

def test_launch():
    """Test launching the application."""
    print("Testing SAFLA HuggingFace Space launch...")
    
    try:
        # Create app
        app = SAFLAApp()
        print("✅ App created successfully")
        
        # Create interface
        interface = app.create_interface()
        print("✅ Interface created successfully")
        
        # Test configuration
        config = app._get_current_config()
        print(f"✅ Configuration loaded: {config['environment']} environment")
        
        # Test system status (async)
        import asyncio
        status = asyncio.run(app._get_system_status())
        print(f"✅ System status: {status}")
        
        print("\n🎉 All tests passed! Application is ready to launch.")
        print("\nTo launch the app, run: python app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_launch()
    sys.exit(0 if success else 1)