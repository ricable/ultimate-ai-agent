#!/bin/bash

# Advanced automation demo
# Shows complex multi-step interactions and real-world use cases

# Time sheet automation
echo "Demonstrating timesheet automation..."
auto-browser easy --interactive "https://workday.com" "Login with username $USER_EMAIL, go to time sheet, and enter 8 hours for today under project 'Development'"

# Social media posting
echo -e "\nDemonstrating social media automation..."
auto-browser easy --interactive "https://twitter.com" "Login with username $TWITTER_USER, create a new post saying 'Just released a new version of auto-browser! Check it out at https://github.com/ruvnet/auto-browser 🚀', and publish it"

# Calendar management
echo -e "\nDemonstrating calendar automation..."
auto-browser easy --interactive "https://calendar.google.com" "Login, create a new meeting called 'Team Sync' for next Tuesday at 2pm, invite team@company.com, add video call, and send invites"

# Price comparison
echo -e "\nDemonstrating price comparison..."
auto-browser easy -v -r "https://amazon.com" "Search for iPhone 15 Pro, extract price, then go to bestbuy.com and get their price for comparison"

# Research workflow
echo -e "\nDemonstrating research automation..."
auto-browser easy -v -r "https://scholar.google.com" "Search for 'LLM automation' papers from 2023, extract top 5 paper titles and citations, then get their abstracts"

# Report generation
echo -e "\nDemonstrating report automation..."
auto-browser easy -v -r "https://analytics.google.com" "Login, get last week's traffic stats, export top pages report, then go to mailchimp.com and create draft newsletter with these stats"
