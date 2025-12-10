#!/bin/bash

# NPM Package Statistics Dashboard
# Get comprehensive stats for any npm package

set -e

PACKAGE=${1:-"agentic-flow"}

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         NPM Package Statistics Dashboard                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Package: $PACKAGE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "⚠️  jq not found, installing..."
    npm install -g jq 2>/dev/null || echo "Manual install required: apt-get install jq"
fi

# 1. Basic Package Info (No login required)
echo "📦 Package Information:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
npm view "$PACKAGE" name version description homepage repository.url --json 2>/dev/null | jq -r '
  "Name:        \(.name)",
  "Version:     \(.version)",
  "Description: \(.description)",
  "Homepage:    \(.homepage)",
  "Repository:  \(.["repository.url"])"
' || echo "Unable to fetch package info"

echo ""
echo "📅 Publishing Information:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
npm view "$PACKAGE" time.created time.modified --json 2>/dev/null | jq -r '
  "Created:  \(.["time.created"])",
  "Modified: \(.["time.modified"])"
' || echo "Unable to fetch dates"

echo ""
echo "👤 Maintainers:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
npm view "$PACKAGE" maintainers --json 2>/dev/null | jq -r '.[] | "  • \(.name) <\(.email)>"' || echo "Unable to fetch maintainers"

echo ""
echo "📊 Download Statistics (NPM API):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Last Day
LAST_DAY=$(curl -s "https://api.npmjs.org/downloads/point/last-day/$PACKAGE" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "Last Day:   $(echo $LAST_DAY | jq -r '.downloads // "N/A"')"
else
    echo "Last Day:   Offline"
fi

# Last Week
LAST_WEEK=$(curl -s "https://api.npmjs.org/downloads/point/last-week/$PACKAGE" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "Last Week:  $(echo $LAST_WEEK | jq -r '.downloads // "N/A"')"
else
    echo "Last Week:  Offline"
fi

# Last Month
LAST_MONTH=$(curl -s "https://api.npmjs.org/downloads/point/last-month/$PACKAGE" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "Last Month: $(echo $LAST_MONTH | jq -r '.downloads // "N/A"')"
else
    echo "Last Month: Offline"
fi

# Last Year
LAST_YEAR=$(curl -s "https://api.npmjs.org/downloads/point/last-year/$PACKAGE" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "Last Year:  $(echo $LAST_YEAR | jq -r '.downloads // "N/A"')"
else
    echo "Last Year:  Offline"
fi

echo ""
echo "📈 Version History (Last 10):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
npm view "$PACKAGE" versions --json 2>/dev/null | jq -r '.[-10:][]' | while read -r version; do
    echo "  • $version"
done || echo "Unable to fetch versions"

echo ""
echo "🔗 Useful Links:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "NPM Page:      https://www.npmjs.com/package/$PACKAGE"
echo "Download Stats: https://npmcharts.com/compare/$PACKAGE"
echo "NPM Trends:    https://npmtrends.com/$PACKAGE"
echo "Bundlephobia:  https://bundlephobia.com/package/$PACKAGE"
echo "Skypack:       https://www.skypack.dev/view/$PACKAGE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "To get stats for your own packages, login first:"
echo "  npm login"
echo "  npm profile get"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
