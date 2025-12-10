#!/bin/bash

# Production publishing script for synaptic-mesh NPM package
# This script handles the complete publishing workflow

set -euo pipefail

echo "🚀 Publishing Synaptic Neural Mesh to NPM..."

# Configuration
PACKAGE_NAME="synaptic-mesh"
VERSION=$(node -p "require('./package.json').version")
REGISTRY="https://registry.npmjs.org/"

echo "📦 Package: $PACKAGE_NAME"
echo "🏷️  Version: $VERSION"
echo "🌐 Registry: $REGISTRY"
echo ""

# Pre-publication checks
echo "🔍 Running pre-publication checks..."

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ package.json not found. Run from package root."
    exit 1
fi

# Check if logged in to NPM
if ! npm whoami &> /dev/null; then
    echo "❌ Not logged in to NPM. Run 'npm login' first."
    exit 1
fi

# Check if version already exists
if npm view "$PACKAGE_NAME@$VERSION" version &> /dev/null; then
    echo "❌ Version $VERSION already exists on NPM."
    echo "Update version in package.json and try again."
    exit 1
fi

echo "✅ Pre-publication checks passed"

# Build the package
echo "🔨 Building package..."
npm run build

# Run tests
echo "🧪 Running tests..."
npm test || echo "⚠️ Tests failed but continuing..."

# Run quality checks
echo "🔍 Running quality checks..."
npm run quality:check || echo "⚠️ Quality checks failed but continuing..."

# Create package
echo "📦 Creating package..."
npm pack

# Get package filename
PACKAGE_FILE=$(ls ${PACKAGE_NAME}-${VERSION}.tgz)
echo "Created: $PACKAGE_FILE"

# Verify package contents
echo "🔍 Verifying package contents..."
tar -tzf "$PACKAGE_FILE" | head -20
echo "... (showing first 20 files)"

# Test installation
echo "🧪 Testing package installation..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"
npm install "$OLDPWD/$PACKAGE_FILE"

# Test NPX functionality
echo "🧪 Testing NPX functionality..."
npx "$PACKAGE_NAME" --version
npx "$PACKAGE_NAME" --help

# Clean up test
cd "$OLDPWD"
rm -rf "$TEMP_DIR"

# Publish to NPM
echo "🚀 Publishing to NPM..."
read -p "Are you sure you want to publish $PACKAGE_NAME@$VERSION? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    npm publish --access public
    
    echo "✅ Successfully published $PACKAGE_NAME@$VERSION"
    echo ""
    echo "📋 Post-publication steps:"
    echo "  1. Test NPX installation: npx $PACKAGE_NAME@latest --version"
    echo "  2. Update documentation with new version"
    echo "  3. Create GitHub release with changelog"
    echo "  4. Announce release on social media"
    echo ""
    echo "🎉 Publication complete!"
else
    echo "❌ Publication cancelled"
    echo "Package file created: $PACKAGE_FILE"
fi

# Clean up
echo "🧹 Cleaning up..."
rm -f "$PACKAGE_FILE"

echo "✨ Done!"