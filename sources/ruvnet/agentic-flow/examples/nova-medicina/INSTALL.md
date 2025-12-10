# Nova Medicina Installation Guide

## Quick Install

```bash
# Global installation (recommended for CLI usage)
npm install -g nova-medicina

# Local installation (for programmatic usage)
npm install nova-medicina

# Using npx (no installation required)
npx nova-medicina analyze "headache" --age 35
```

## Prerequisites

- **Node.js**: Version 18.0.0 or higher
- **npm**: Version 8.0.0 or higher (comes with Node.js)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: At least 512MB RAM available
- **Disk Space**: 100MB for installation

## Verify Installation

```bash
# Check Node.js version
node --version  # Should show v18.0.0 or higher

# Check npm version
npm --version   # Should show 8.0.0 or higher

# Install nova-medicina
npm install -g nova-medicina

# Verify installation
nova-medicina --version
nova-medicina --help
```

## Configuration

### 1. API Keys (Optional)

For enhanced verification features, configure API keys:

```bash
# Set via command
nova-medicina config --set apiKey=your-api-key-here

# Or via environment variable
export NOVA_MEDICINA_API_KEY="your-api-key-here"

# Or via config file
echo '{"apiKey": "your-api-key-here"}' > ~/.nova-medicina/config.json
```

### 2. Provider Integration (Healthcare Providers Only)

```bash
# Configure provider credentials
nova-medicina config --set providerId=your-provider-id
nova-medicina config --set providerApiKey=your-provider-api-key

# Test provider connection
nova-medicina provider --search "cardiology"
```

### 3. AgentDB Configuration (Advanced)

```bash
# Configure AgentDB for learning
nova-medicina config --set agentdb.enabled=true
nova-medicina config --set agentdb.path=~/.nova-medicina/agentdb
```

## First-Time Setup

### Interactive Setup Wizard

```bash
nova-medicina config --setup
```

This will guide you through:
1. ✅ API key configuration
2. ✅ Provider integration (optional)
3. ✅ Privacy preferences
4. ✅ Notification settings
5. ✅ AgentDB learning configuration

### Manual Configuration

Create `~/.nova-medicina/config.json`:

```json
{
  "apiKey": "your-api-key",
  "verification": {
    "level": "standard",
    "confidenceThreshold": 0.75
  },
  "provider": {
    "enabled": false,
    "id": null,
    "apiKey": null
  },
  "agentdb": {
    "enabled": true,
    "path": "~/.nova-medicina/agentdb"
  },
  "privacy": {
    "shareAnonymousData": false,
    "storageLocation": "local"
  }
}
```

## Usage Examples

### Basic Usage

```bash
# Simple symptom analysis
nova-medicina analyze "fever and cough"

# With patient details
nova-medicina analyze "chest pain" --age 45 --gender M

# Interactive mode
nova-medicina analyze --interactive
```

### Verification

```bash
# Verify a diagnosis
nova-medicina verify --diagnosis "Type 2 Diabetes"

# Verify treatment recommendation
nova-medicina verify --treatment "Metformin 500mg twice daily"
```

### Provider Search

```bash
# Find providers by specialty
nova-medicina provider --search "cardiology" --location "New York"

# Display provider information
nova-medicina provider --info "provider-12345"
```

## Troubleshooting

### Installation Issues

**Error: "EACCES: permission denied"**
```bash
# Solution: Use sudo or configure npm prefix
sudo npm install -g nova-medicina

# Or configure npm to install globally without sudo
npm config set prefix ~/.npm-global
export PATH=~/.npm-global/bin:$PATH
npm install -g nova-medicina
```

**Error: "Node version not supported"**
```bash
# Solution: Update Node.js
# Visit https://nodejs.org/ to download latest version
# Or use nvm (Node Version Manager)
nvm install 18
nvm use 18
```

### Runtime Issues

**Error: "API key not configured"**
```bash
# Solution: Set API key
nova-medicina config --set apiKey=your-key
```

**Error: "AgentDB connection failed"**
```bash
# Solution: Reset AgentDB
rm -rf ~/.nova-medicina/agentdb
nova-medicina config --set agentdb.enabled=true
```

### Getting Help

```bash
# View all available commands
nova-medicina --help

# Command-specific help
nova-medicina analyze --help
nova-medicina verify --help
nova-medicina provider --help

# Interactive tutorial
nova-medicina tutorial
```

## Upgrading

```bash
# Check current version
nova-medicina --version

# Update to latest version
npm update -g nova-medicina

# Or reinstall
npm uninstall -g nova-medicina
npm install -g nova-medicina
```

## Uninstalling

```bash
# Uninstall package
npm uninstall -g nova-medicina

# Remove configuration and data
rm -rf ~/.nova-medicina
```

## Development Installation

For contributors and developers:

```bash
# Clone repository
git clone https://github.com/ruvnet/nova-medicina.git
cd nova-medicina

# Install dependencies
npm install

# Build TypeScript
npm run build

# Run tests
npm test

# Link for local development
npm link

# Now use nova-medicina command
nova-medicina --help
```

## Platform-Specific Notes

### macOS

```bash
# Install via Homebrew (if available)
brew install node
npm install -g nova-medicina
```

### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nodejs npm
npm install -g nova-medicina

# Fedora/RHEL
sudo dnf install nodejs npm
npm install -g nova-medicina
```

### Windows

```powershell
# Install via Chocolatey
choco install nodejs
npm install -g nova-medicina

# Or download Node.js from https://nodejs.org/
# Then run in PowerShell:
npm install -g nova-medicina
```

## Docker Installation (Advanced)

```bash
# Pull Docker image
docker pull ruvnet/nova-medicina:latest

# Run container
docker run -it --rm \
  -v ~/.nova-medicina:/root/.nova-medicina \
  ruvnet/nova-medicina:latest \
  nova-medicina analyze "fever"
```

## Next Steps

1. ✅ **Read Documentation**: `/docs/PATIENT_GUIDE.md` or `/docs/PROVIDER_GUIDE.md`
2. ✅ **Run Tutorial**: `nova-medicina tutorial`
3. ✅ **Try Examples**: Check `/examples` directory
4. ✅ **Configure Settings**: `nova-medicina config --setup`

## Support

- **Documentation**: https://github.com/ruvnet/nova-medicina/docs
- **Issues**: https://github.com/ruvnet/nova-medicina/issues
- **Community**: https://github.com/ruvnet/nova-medicina/discussions

---

**Created by**: ruv (github.com/ruvnet, ruv.io)
**License**: MIT
**Version**: 1.0.0
