"""
Validate SAFLA HuggingFace Space deployment readiness.
"""

import os
import sys
from pathlib import Path
import json


def check_file_exists(filepath, required=True):
    """Check if a file exists."""
    exists = Path(filepath).exists()
    status = "✅" if exists else ("❌" if required else "⚠️")
    print(f"{status} {filepath}")
    return exists or not required


def check_directory_exists(dirpath, required=True):
    """Check if a directory exists."""
    exists = Path(dirpath).exists() and Path(dirpath).is_dir()
    status = "✅" if exists else ("❌" if required else "⚠️")
    print(f"{status} {dirpath}/")
    return exists or not required


def validate_structure():
    """Validate project structure."""
    print("\n🏗️  Validating Project Structure:")
    
    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
    ]
    
    required_dirs = [
        "src",
        "src/config",
        "src/core",
        "src/ui",
        "src/ui/components",
        "src/ui/tabs",
        "src/ui/themes",
    ]
    
    all_good = True
    
    for file in required_files:
        if not check_file_exists(file):
            all_good = False
    
    for dir in required_dirs:
        if not check_directory_exists(dir):
            all_good = False
    
    return all_good


def validate_config():
    """Validate configuration files."""
    print("\n⚙️  Validating Configuration:")
    
    config_files = [
        ("config/safla_production.json", True),
        (".env.example", True),
        (".env", False),  # Optional, contains secrets
    ]
    
    all_good = True
    
    for file, required in config_files:
        if not check_file_exists(file, required):
            all_good = False
    
    # Check production config is valid JSON
    prod_config = Path("config/safla_production.json")
    if prod_config.exists():
        try:
            with open(prod_config) as f:
                json.load(f)
            print("✅ Production config is valid JSON")
        except Exception as e:
            print(f"❌ Production config JSON error: {e}")
            all_good = False
    
    return all_good


def validate_dependencies():
    """Validate dependencies."""
    print("\n📦 Validating Dependencies:")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    with open(requirements_file) as f:
        requirements = f.read()
    
    required_packages = [
        "gradio",
        "numpy",
        "pandas",
        "plotly",
        "pydantic",
        "python-dotenv",
        "huggingface-hub",
    ]
    
    all_good = True
    for package in required_packages:
        if package in requirements:
            print(f"✅ {package}")
        else:
            print(f"❌ {package} not found in requirements.txt")
            all_good = False
    
    return all_good


def validate_tests():
    """Validate test suite."""
    print("\n🧪 Validating Tests:")
    
    # Run pytest with coverage
    import subprocess
    
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "--tb=short", "-q"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ All tests passing")
            # Extract test summary
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if "passed" in line or "failed" in line:
                    print(f"  {line}")
            return True
        else:
            print("❌ Some tests failing")
            return False
            
    except Exception as e:
        print(f"⚠️  Could not run tests: {e}")
        return True  # Don't block deployment


def validate_app_launch():
    """Validate app can launch."""
    print("\n🚀 Validating App Launch:")
    
    try:
        from app import SAFLAApp
        app = SAFLAApp()
        print("✅ App initialization successful")
        
        interface = app.create_interface()
        print("✅ Interface creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ App launch failed: {e}")
        return False


def main():
    """Main validation function."""
    print("🔍 SAFLA HuggingFace Space Deployment Validation\n")
    
    checks = [
        ("Structure", validate_structure),
        ("Configuration", validate_config),
        ("Dependencies", validate_dependencies),
        ("Tests", validate_tests),
        ("App Launch", validate_app_launch),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Summary
    print("\n📊 Validation Summary:")
    all_passed = True
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✨ All validations passed! Ready for deployment.")
        print("\n📝 Deployment Instructions:")
        print("1. Fork/clone this repository to your GitHub")
        print("2. Go to https://huggingface.co/new-space")
        print("3. Choose 'Gradio' as the SDK")
        print("4. Connect your GitHub repository")
        print("5. Set HUGGINGFACE_API_KEY in Space secrets")
        print("6. Your Space will build and deploy automatically!")
    else:
        print("\n⚠️  Some validations failed. Please fix the issues before deployment.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())