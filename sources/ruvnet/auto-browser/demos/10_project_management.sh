#!/bin/bash

# Project Management Workflow Demo
# Shows automated project setup and team coordination

# Create templates
echo "Creating project management templates..."
auto-browser create-template "https://github.com" --name github --description "GitHub project setup"
auto-browser create-template "https://jira.atlassian.com" --name jira --description "Jira workflow"
auto-browser create-template "https://slack.com" --name slack --description "Team communication"
auto-browser create-template "https://confluence.atlassian.com" --name confluence --description "Documentation"

# Initial project setup
echo -e "\nSetting up GitHub repository..."
auto-browser easy --interactive -v --site github "https://github.com" "Login, then:
1. Create new repository 'ai-automation-project'
2. Initialize with README and .gitignore for Python
3. Add MIT license
4. Create develop and staging branches
5. Set up branch protection rules
6. Enable discussions and wiki"

# Issue tracking setup
echo -e "\nSetting up Jira project..."
auto-browser easy --interactive -v --site jira "https://jira.atlassian.com" "Login, create new project:
1. Name: 'AI Automation'
2. Set up Kanban board
3. Create epics:
   - Core Development
   - Documentation
   - Testing
   - DevOps
4. Configure workflows and permissions"

# Documentation setup
echo -e "\nSetting up Confluence space..."
auto-browser easy --interactive -v --site confluence "https://confluence.atlassian.com" "Login, create new space:
1. Name: 'AI Automation Docs'
2. Create template pages:
   - Getting Started
   - Development Guide
   - API Documentation
   - Deployment Guide
3. Set up page hierarchy and permissions"

# Team communication
echo -e "\nSetting up Slack workspace..."
auto-browser easy --interactive -v --site slack "https://slack.com" "Login, then:
1. Create channels:
   - #ai-automation-general
   - #ai-automation-dev
   - #ai-automation-alerts
2. Invite team members
3. Set channel topics and purposes
4. Pin important links"

# CI/CD setup
echo -e "\nSetting up GitHub Actions..."
auto-browser easy --interactive -v --site github "https://github.com/ruvnet/ai-automation-project" "Create workflows:
1. Python testing workflow
2. Linting and code quality
3. Documentation generation
4. Container builds
5. Deployment pipeline"

# Integration setup
echo -e "\nSetting up integrations..."
auto-browser easy --interactive -v "https://github.com" "Set up integrations:
1. Connect GitHub with Jira
2. Add Slack notifications
3. Configure Confluence sync
4. Set up status checks"

# Project board setup
echo -e "\nSetting up project tracking..."
auto-browser easy --interactive -v --site github "https://github.com/ruvnet/ai-automation-project" "Create project board:
1. Set up automated kanban
2. Create milestone: 'MVP Release'
3. Add initial issues:
   - Project setup
   - Core features
   - Documentation
   - Testing framework"

# Team onboarding
echo -e "\nPreparing team onboarding..."
auto-browser easy --interactive -v --site confluence "https://confluence.atlassian.com" "Create onboarding guide:
1. Repository setup
2. Development workflow
3. Testing procedures
4. Documentation guidelines
5. Deployment process"
