"""
Deploy SAFLA HuggingFace Space to HuggingFace Hub.
"""

import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SPACE_NAME = "safla-demo"
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
USERNAME = None  # Will be fetched from API


def get_username():
    """Get HuggingFace username from API."""
    api = HfApi(token=HF_TOKEN)
    try:
        user_info = api.whoami()
        return user_info["name"]
    except Exception as e:
        print(f"❌ Failed to get username: {e}")
        print("Make sure your HUGGINGFACE_API_KEY is valid.")
        return None


def prepare_deployment():
    """Prepare files for deployment."""
    print("🔧 Preparing deployment files...")
    
    # Create deployment directory
    deploy_dir = Path("deploy_temp")
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()
    
    # Files to include in deployment
    files_to_copy = [
        "app.py",
        "requirements.txt",
        "README.md",
        ".env.example",
        "pytest.ini",
    ]
    
    # Directories to include
    dirs_to_copy = [
        "src",
        "assets",
        "config",
    ]
    
    # Copy files
    for file in files_to_copy:
        if Path(file).exists():
            shutil.copy2(file, deploy_dir)
            print(f"  ✅ Copied {file}")
    
    # Copy directories
    for dir_name in dirs_to_copy:
        if Path(dir_name).exists():
            shutil.copytree(dir_name, deploy_dir / dir_name)
            print(f"  ✅ Copied {dir_name}/")
    
    # Create a production .env file (without actual secrets)
    env_content = """# Production environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# These will be set as HuggingFace Spaces secrets
# HUGGINGFACE_API_KEY=
# SAFLA_CONFIG_PATH=
"""
    (deploy_dir / ".env").write_text(env_content)
    
    return deploy_dir


def create_space(username, space_name):
    """Create or update HuggingFace Space."""
    api = HfApi(token=HF_TOKEN)
    repo_id = f"{username}/{space_name}"
    
    try:
        # Try to create the space
        print(f"📦 Creating space: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True
        )
        print(f"  ✅ Space created/updated: {repo_id}")
        return repo_id
    except Exception as e:
        print(f"  ❌ Failed to create space: {e}")
        return None


def upload_to_space(repo_id, deploy_dir):
    """Upload files to HuggingFace Space."""
    api = HfApi(token=HF_TOKEN)
    
    try:
        print(f"📤 Uploading files to {repo_id}...")
        api.upload_folder(
            folder_path=str(deploy_dir),
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=[
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
                "tests/",
                "*.log",
                ".git/",
                "deploy_temp/"
            ]
        )
        print(f"  ✅ Files uploaded successfully!")
        return True
    except Exception as e:
        print(f"  ❌ Failed to upload files: {e}")
        return False


def set_space_secrets(repo_id):
    """Set environment variables as Space secrets."""
    api = HfApi(token=HF_TOKEN)
    
    print(f"🔐 Setting space secrets...")
    
    secrets = {
        "HUGGINGFACE_API_KEY": HF_TOKEN,
        "SAFLA_CONFIG_PATH": "./config/safla_production.json"
    }
    
    for key, value in secrets.items():
        try:
            # Note: HuggingFace API doesn't directly support setting secrets
            # This would need to be done manually in the UI
            print(f"  ⚠️  Please set {key} manually in Space settings")
        except Exception as e:
            print(f"  ❌ Failed to set {key}: {e}")


def main():
    """Main deployment function."""
    print("🚀 SAFLA HuggingFace Space Deployment Tool\n")
    
    # Check for API key
    if not HF_TOKEN:
        print("❌ HUGGINGFACE_API_KEY not found in environment variables.")
        print("Please set it in your .env file or export it.")
        return 1
    
    # Get username
    username = get_username()
    if not username:
        return 1
    
    print(f"👤 Deploying as user: {username}\n")
    
    # Prepare deployment
    deploy_dir = prepare_deployment()
    
    # Create space
    repo_id = create_space(username, SPACE_NAME)
    if not repo_id:
        return 1
    
    # Upload files
    if not upload_to_space(repo_id, deploy_dir):
        return 1
    
    # Set secrets (manual step required)
    set_space_secrets(repo_id)
    
    # Cleanup
    shutil.rmtree(deploy_dir)
    
    print(f"\n✨ Deployment complete!")
    print(f"🌐 Your Space URL: https://huggingface.co/spaces/{repo_id}")
    print(f"\n⚠️  Don't forget to:")
    print(f"  1. Set the HUGGINGFACE_API_KEY secret in Space settings")
    print(f"  2. Check the Space build logs")
    print(f"  3. Test the live demo")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())