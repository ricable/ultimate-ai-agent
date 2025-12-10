#!/usr/bin/env python3
"""
Deployment script for SAFLA HuggingFace Space.
Automates the deployment process to HuggingFace Spaces.
"""

import os
import sys
import subprocess
import shutil
import json
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HuggingFaceDeployer:
    """Manages deployment to HuggingFace Spaces."""
    
    def __init__(self, space_name: str, token: str = None):
        """
        Initialize deployer.
        
        Args:
            space_name: HuggingFace space name (username/space-name)
            token: HuggingFace API token (optional, uses HF_TOKEN env var)
        """
        self.space_name = space_name
        self.token = token or os.environ.get("HF_TOKEN")
        self.project_root = Path(__file__).parent.parent
        
        if not self.token:
            logger.warning("No HuggingFace token provided. Set HF_TOKEN environment variable.")
    
    def check_requirements(self) -> bool:
        """Check if all requirements are met for deployment."""
        logger.info("Checking deployment requirements...")
        
        # Check for required files
        required_files = [
            self.project_root / "app.py",
            self.project_root / "requirements.txt",
            self.project_root / "README.md"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required file missing: {file_path}")
                return False
        
        # Check for Git
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.error("Git is not installed")
            return False
        
        # Check for huggingface-cli
        try:
            subprocess.run(["huggingface-cli", "--version"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.error("huggingface-cli is not installed. Run: pip install huggingface-hub")
            return False
        
        logger.info("All requirements met")
        return True
    
    def run_tests(self) -> bool:
        """Run tests before deployment."""
        logger.info("Running tests...")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-v"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Tests failed:\n{result.stdout}\n{result.stderr}")
                return False
            
            logger.info("All tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False
    
    def optimize_for_deployment(self) -> None:
        """Optimize the application for deployment."""
        logger.info("Optimizing for deployment...")
        
        # Create .env.example if .env exists
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"
        
        if env_file.exists() and not env_example.exists():
            with open(env_file, 'r') as f:
                lines = f.readlines()
            
            with open(env_example, 'w') as f:
                for line in lines:
                    if '=' in line:
                        key = line.split('=')[0].strip()
                        f.write(f"{key}=your_{key.lower()}_here\n")
                    else:
                        f.write(line)
            
            logger.info("Created .env.example from .env")
        
        # Update requirements.txt to remove local paths
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            with open(req_file, 'r') as f:
                lines = f.readlines()
            
            filtered_lines = []
            for line in lines:
                # Skip local package installations
                if not line.strip().startswith("-e"):
                    filtered_lines.append(line)
            
            with open(req_file, 'w') as f:
                f.writelines(filtered_lines)
            
            logger.info("Updated requirements.txt for deployment")
    
    def create_deployment_branch(self) -> bool:
        """Create a deployment branch with optimizations."""
        logger.info("Creating deployment branch...")
        
        try:
            # Create new branch
            subprocess.run(
                ["git", "checkout", "-b", "huggingface-deployment"],
                cwd=self.project_root,
                check=True
            )
            
            # Run optimizations
            self.optimize_for_deployment()
            
            # Stage changes
            subprocess.run(
                ["git", "add", "."],
                cwd=self.project_root,
                check=True
            )
            
            # Commit changes
            subprocess.run(
                ["git", "commit", "-m", "Optimize for HuggingFace deployment"],
                cwd=self.project_root,
                check=True
            )
            
            logger.info("Deployment branch created")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating deployment branch: {e}")
            return False
    
    def deploy_to_huggingface(self) -> bool:
        """Deploy to HuggingFace Spaces."""
        logger.info(f"Deploying to HuggingFace Space: {self.space_name}")
        
        if not self.token:
            logger.error("HuggingFace token required for deployment")
            return False
        
        try:
            # Login to HuggingFace
            subprocess.run(
                ["huggingface-cli", "login", "--token", self.token],
                check=True,
                capture_output=True
            )
            
            # Create or update space
            space_url = f"https://huggingface.co/spaces/{self.space_name}"
            
            # Clone or create space repository
            space_dir = Path(f"/tmp/safla-space-{self.space_name.replace('/', '-')}")
            
            if space_dir.exists():
                shutil.rmtree(space_dir)
            
            # Try to clone existing space
            try:
                subprocess.run(
                    ["git", "clone", f"{space_url}.git", str(space_dir)],
                    check=True,
                    capture_output=True
                )
                logger.info("Cloned existing space")
            except subprocess.CalledProcessError:
                # Create new space
                space_dir.mkdir(parents=True)
                subprocess.run(
                    ["git", "init"],
                    cwd=space_dir,
                    check=True
                )
                logger.info("Created new space repository")
            
            # Copy files
            files_to_copy = [
                "app.py",
                "requirements.txt",
                "README.md",
                ".gitignore"
            ]
            
            dirs_to_copy = [
                "src",
                "assets",
                "docs"
            ]
            
            for file_name in files_to_copy:
                src_file = self.project_root / file_name
                if src_file.exists():
                    shutil.copy2(src_file, space_dir / file_name)
            
            for dir_name in dirs_to_copy:
                src_dir = self.project_root / dir_name
                if src_dir.exists():
                    dst_dir = space_dir / dir_name
                    if dst_dir.exists():
                        shutil.rmtree(dst_dir)
                    shutil.copytree(src_dir, dst_dir)
            
            # Add and commit
            subprocess.run(
                ["git", "add", "."],
                cwd=space_dir,
                check=True
            )
            
            subprocess.run(
                ["git", "commit", "-m", "Update SAFLA HuggingFace Space"],
                cwd=space_dir,
                check=True
            )
            
            # Set remote and push
            subprocess.run(
                ["git", "remote", "add", "origin", f"{space_url}.git"],
                cwd=space_dir,
                check=True,
                capture_output=True
            )
            
            subprocess.run(
                ["git", "push", "--force", "origin", "main"],
                cwd=space_dir,
                check=True
            )
            
            logger.info(f"Successfully deployed to {space_url}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Deployment failed: {e}")
            return False
        finally:
            # Cleanup
            if 'space_dir' in locals() and space_dir.exists():
                shutil.rmtree(space_dir)
    
    def verify_deployment(self) -> bool:
        """Verify the deployment was successful."""
        logger.info("Verifying deployment...")
        
        import requests
        import time
        
        space_url = f"https://huggingface.co/spaces/{self.space_name}"
        
        # Wait for space to be ready
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(space_url, timeout=10)
                if response.status_code == 200:
                    logger.info("Space is accessible")
                    return True
            except requests.RequestException:
                pass
            
            logger.info(f"Waiting for space to be ready... ({attempt + 1}/{max_attempts})")
            time.sleep(10)
        
        logger.error("Space verification failed")
        return False
    
    def deploy(self, skip_tests: bool = False) -> bool:
        """
        Run full deployment process.
        
        Args:
            skip_tests: Whether to skip running tests
            
        Returns:
            Success status
        """
        # Check requirements
        if not self.check_requirements():
            return False
        
        # Run tests unless skipped
        if not skip_tests and not self.run_tests():
            logger.error("Tests failed. Use --skip-tests to bypass.")
            return False
        
        # Create deployment branch
        if not self.create_deployment_branch():
            return False
        
        # Deploy to HuggingFace
        if not self.deploy_to_huggingface():
            return False
        
        # Verify deployment
        if self.verify_deployment():
            logger.info("Deployment completed successfully!")
            logger.info(f"Visit your space at: https://huggingface.co/spaces/{self.space_name}")
            return True
        
        return False


def main():
    """Main entry point for deployment script."""
    parser = argparse.ArgumentParser(
        description="Deploy SAFLA to HuggingFace Spaces"
    )
    parser.add_argument(
        "space_name",
        help="HuggingFace space name (format: username/space-name)"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace API token (or set HF_TOKEN env var)",
        default=None
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests before deployment"
    )
    
    args = parser.parse_args()
    
    # Validate space name format
    if "/" not in args.space_name:
        logger.error("Space name must be in format: username/space-name")
        sys.exit(1)
    
    # Create deployer and run deployment
    deployer = HuggingFaceDeployer(args.space_name, args.token)
    
    if deployer.deploy(skip_tests=args.skip_tests):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()