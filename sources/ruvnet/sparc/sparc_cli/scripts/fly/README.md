# SPARC CLI Fly.io Deployment

This directory contains scripts for deploying SPARC CLI to Fly.io. The deployment creates a dedicated VM with SPARC CLI installed and accessible via SSH.

## Prerequisites

- Git installed on your system
- Linux or macOS (Windows users should use WSL)
- For macOS users: Homebrew installed
- A Fly.io account (sign up at https://fly.io)

## Quick Start

1. Copy and edit configuration:
   ```bash
   cp config.template.sh config.sh
   # Edit config.sh with your settings
   ```

2. Run the installation script:
   ```bash
   ./install.sh
   ```
   This will:
   - Install the Fly.io CLI
   - Log you into Fly.io
   - Create your application
   - Set up environment variables

3. Deploy your application:
   ```bash
   ./deploy.sh
   ```
   This will:
   - Create a VM with SPARC CLI installed
   - Configure SSH access
   - Set up environment variables

4. Access your deployment:
   ```bash
   ./access.sh
   ```
   Choose from multiple access options:
   - Chat Mode: Interactive chat with SPARC CLI
   - Research Mode: Research-only mode for information gathering
   - Cowboy Mode: Skip command confirmations for faster execution
   - Manual Console: Direct SSH access for custom commands
   - View Logs: Monitor application logs

## Configuration

The `config.sh` file (created from `config.template.sh`) contains all deployment settings:

- `FLY_APP_NAME`: Your Fly.io application name
- `FLY_REGION`: Deployment region (e.g., "lax", "sfo", "iad")
- `FLY_ORG`: Your Fly.io organization
- `GIT_REPO_URL`: GitHub repository URL (optional, can be entered during deployment)
- `GIT_BRANCH`: Repository branch (defaults to "main")
- API keys for SPARC CLI (ANTHROPIC_API_KEY required)
- Deployment machine configuration

## Scripts

### install.sh
- Installs Fly.io CLI
- Sets up your application
- Configures environment variables

### deploy.sh
- Prompts for GitHub repository if not configured
- Clones and deploys your application
- Creates Fly.io configuration

### access.sh
- Provides interactive access to your deployment
- Shows logs and application status
- Opens web interface

## Security Notes

- `config.sh` contains sensitive information and is excluded from git
- API keys are stored securely as Fly.io secrets
- Each deployment creates a fresh clone of your repository

## Troubleshooting

1. If installation fails:
   - Check your Fly.io credentials
   - Ensure you have necessary permissions
   - Verify your API keys

2. If deployment fails:
   - Check the GitHub repository URL and branch
   - Verify the repository contains sparc_cli directory
   - Check Fly.io resource limits

3. If access fails:
   - Ensure the application is deployed
   - Check Fly.io status
   - Verify your network connection

For more help, visit:
- Fly.io Documentation: https://fly.io/docs/
- SPARC CLI Documentation: https://github.com/yourusername/sparc
