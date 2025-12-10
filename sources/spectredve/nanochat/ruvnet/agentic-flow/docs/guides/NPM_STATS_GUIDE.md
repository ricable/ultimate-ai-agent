# NPM Statistics Tools

Complete guide for checking npm package and user statistics via CLI.

## Quick Reference

### Check Package Stats (No Login Required)

```bash
# Basic package info
npm view <package-name>

# Download stats using our tool
./scripts/npm-stats.sh <package-name>

# User's all packages
node scripts/npm-user-stats.js <username>
```

## 1. Using NPM CLI (Without Login)

### Get Package Information
```bash
# Basic info
npm view agentic-flow

# Specific fields
npm view agentic-flow name version description

# All versions
npm view agentic-flow versions

# As JSON
npm view agentic-flow --json

# Package metadata
npm info agentic-flow
```

### Get Download Stats via NPM API
```bash
# Last day
curl https://api.npmjs.org/downloads/point/last-day/agentic-flow

# Last week
curl https://api.npmjs.org/downloads/point/last-week/agentic-flow

# Last month
curl https://api.npmjs.org/downloads/point/last-month/agentic-flow

# Last year
curl https://api.npmjs.org/downloads/point/last-year/agentic-flow

# Date range
curl https://api.npmjs.org/downloads/range/2024-01-01:2024-12-31/agentic-flow
```

## 2. Using NPM CLI (With Login)

### Login to NPM
```bash
# Interactive login
npm login

# Or use adduser (same as login)
npm adduser

# With specific registry
npm login --registry=https://registry.npmjs.org/

# Check if logged in
npm whoami
```

### After Login - Get Your Stats
```bash
# Your profile info
npm profile get

# Your packages
npm profile get packages

# Set profile fields
npm profile set <key> <value>
```

### Programmatic Login (CI/CD)
```bash
# Using auth token
npm config set //registry.npmjs.org/:_authToken $NPM_TOKEN

# Or set in .npmrc
echo "//registry.npmjs.org/:_authToken=$NPM_TOKEN" >> ~/.npmrc
```

## 3. Using Our Custom Tools

### Tool 1: Package Stats Dashboard
```bash
# Make executable
chmod +x scripts/npm-stats.sh

# Get stats for any package
./scripts/npm-stats.sh agentic-flow

# Or specify package
./scripts/npm-stats.sh express
```

**Output includes:**
- Package information
- Publishing dates
- Maintainers
- Download statistics (day/week/month/year)
- Version history
- Useful links

### Tool 2: User Package Statistics
```bash
# Make executable
chmod +x scripts/npm-user-stats.js

# Get all packages for a user
node scripts/npm-user-stats.js ruvnet

# Or use environment variable
NPM_USERNAME=ruvnet node scripts/npm-user-stats.js
```

**Output includes:**
- All packages by user
- Download stats for each
- Total statistics
- Top 5 packages
- Profile links

## 4. Alternative Tools

### Using npx (No Installation)
```bash
# Download stats
npx npm-stats agentic-flow

# With date range
npx npm-stats agentic-flow --start-date 2024-01-01

# Package details
npx package-stats agentic-flow
```

### Using npm-check-updates
```bash
# Install globally
npm install -g npm-check-updates

# Check outdated packages
ncu

# Check specific package
ncu agentic-flow
```

## 5. Web-Based Tools (No Login)

### NPM Website
- Package page: `https://www.npmjs.com/package/<package-name>`
- User profile: `https://www.npmjs.com/~<username>`
- Search: `https://www.npmjs.com/search?q=<query>`

### Third-Party Analytics

**NPM Charts**
- URL: `https://npmcharts.com/compare/<package-name>`
- Features: Download trends, comparisons

**NPM Trends**
- URL: `https://npmtrends.com/<package-name>`
- Features: Multi-package comparison, GitHub stars

**Bundlephobia**
- URL: `https://bundlephobia.com/package/<package-name>`
- Features: Bundle size, tree-shaking

**Skypack**
- URL: `https://www.skypack.dev/view/<package-name>`
- Features: CDN stats, popularity score

## 6. Advanced: NPM Registry API

### Direct API Calls

```bash
# Package metadata
curl https://registry.npmjs.org/<package-name>

# Specific version
curl https://registry.npmjs.org/<package-name>/<version>

# Search packages
curl https://registry.npmjs.org/-/v1/search?text=<query>

# User's packages
curl https://registry.npmjs.org/-/v1/search?text=maintainer:<username>
```

### Download Statistics API

```bash
# Point stats (specific period)
curl https://api.npmjs.org/downloads/point/last-day/<package>
curl https://api.npmjs.org/downloads/point/last-week/<package>
curl https://api.npmjs.org/downloads/point/last-month/<package>
curl https://api.npmjs.org/downloads/point/last-year/<package>

# Range stats (date range)
curl https://api.npmjs.org/downloads/range/2024-01-01:2024-12-31/<package>

# Bulk package stats
curl https://api.npmjs.org/downloads/point/last-month/package1,package2,package3
```

### Response Format
```json
{
  "downloads": 12345,
  "start": "2024-01-01",
  "end": "2024-01-31",
  "package": "package-name"
}
```

## 7. Example Workflows

### Check Your Package Performance
```bash
#!/bin/bash

PACKAGE="agentic-flow"

echo "📊 Package Performance Report"
echo ""

# Current version
VERSION=$(npm view $PACKAGE version)
echo "Current Version: $VERSION"

# Monthly downloads
DOWNLOADS=$(curl -s https://api.npmjs.org/downloads/point/last-month/$PACKAGE | jq -r '.downloads')
echo "Monthly Downloads: $DOWNLOADS"

# Latest release date
MODIFIED=$(npm view $PACKAGE time.modified)
echo "Last Update: $MODIFIED"

# Dependencies
DEPS=$(npm view $PACKAGE dependencies --json | jq 'length')
echo "Dependencies: $DEPS"
```

### Compare Multiple Packages
```bash
#!/bin/bash

PACKAGES=("express" "fastify" "koa")

echo "📈 Framework Comparison"
echo ""

for pkg in "${PACKAGES[@]}"; do
    downloads=$(curl -s https://api.npmjs.org/downloads/point/last-month/$pkg | jq -r '.downloads')
    version=$(npm view $pkg version)
    echo "$pkg: v$version - $downloads downloads/month"
done
```

### Monitor Your Packages
```bash
#!/bin/bash

# Get all your packages
USERNAME="ruvnet"
node scripts/npm-user-stats.js $USERNAME > report.txt

# Email the report (optional)
# mail -s "Weekly NPM Stats" you@example.com < report.txt
```

## 8. Automation Examples

### GitHub Action for Stats
```yaml
name: NPM Stats Report

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:

jobs:
  stats:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Generate Stats
        run: |
          node scripts/npm-user-stats.js ${{ secrets.NPM_USERNAME }} > stats.md

      - name: Commit Report
        run: |
          git config user.name github-actions
          git config user.email github-actions@github.com
          git add stats.md
          git commit -m "Update NPM stats"
          git push
```

### Daily Stats Script
```bash
#!/bin/bash
# Save as: daily-npm-stats.sh
# Add to crontab: 0 9 * * * /path/to/daily-npm-stats.sh

PACKAGE="agentic-flow"
DATE=$(date +%Y-%m-%d)
LOG_FILE="npm-stats-${DATE}.log"

{
    echo "NPM Stats Report - $DATE"
    echo "================================"
    ./scripts/npm-stats.sh $PACKAGE
} > $LOG_FILE

# Optional: Upload to monitoring service
# curl -X POST https://your-monitoring.com/api/stats -d @$LOG_FILE
```

## 9. Troubleshooting

### Not Logged In
```bash
# Error: need auth
npm login

# Verify login
npm whoami

# Check token
npm config get //registry.npmjs.org/:_authToken
```

### API Rate Limiting
```bash
# Wait between requests
sleep 1

# Use exponential backoff
# Implement in your scripts
```

### Package Not Found
```bash
# Check package name
npm search <partial-name>

# Check registry
npm config get registry

# Try full URL
npm view https://registry.npmjs.org/<package-name>
```

## 10. Best Practices

1. **Cache API Results**: Don't fetch same data repeatedly
2. **Rate Limiting**: Add delays between API calls
3. **Error Handling**: Check for network errors
4. **Token Security**: Never commit NPM tokens
5. **Use Official Tools**: Prefer npm CLI over direct API

## Tools Summary

| Tool | Login Required | Use Case |
|------|----------------|----------|
| `npm view` | No | Package info |
| `npm whoami` | Yes | Check login |
| `npm profile get` | Yes | Your profile |
| `npm-stats.sh` | No | Package stats |
| `npm-user-stats.js` | No | All user packages |
| NPM API | No | Programmatic access |
| Web tools | No | Visual analytics |

## Support

- **NPM Docs**: https://docs.npmjs.com/
- **NPM API**: https://github.com/npm/registry/blob/master/docs/REGISTRY-API.md
- **Download API**: https://github.com/npm/registry/blob/master/docs/download-counts.md

---

**Created**: 2025-10-21
**Location**: `/home/user/agentic-flow/scripts/`
