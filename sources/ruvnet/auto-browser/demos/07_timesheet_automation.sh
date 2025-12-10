#!/bin/bash

# Timesheet Automation Demo
# Shows complex multi-step timesheet management

# Create template for Workday
echo "Creating Workday template..."
auto-browser create-template "https://workday.com" --name workday --description "Timesheet automation"

# Weekly timesheet entry
echo -e "\nFilling out weekly timesheet..."
auto-browser easy --interactive -v "https://workday.com" "Login with username $USER_EMAIL, go to time sheet, and for each day this week enter: 6 hours for project 'Development', 1 hour for 'Team Meeting', and 1 hour for 'Documentation'"

# Submit timesheet with notes
echo -e "\nSubmitting timesheet with details..."
auto-browser easy --interactive -v --site workday "https://workday.com" "Go to current timesheet, add comment 'Completed sprint tasks and documentation', verify total hours are 40, then submit for approval"

# Check approval status
echo -e "\nChecking approval status..."
auto-browser easy -v -r --site workday "https://workday.com" "Go to submitted timesheets and get status of this week's timesheet"

# Generate timesheet report
echo -e "\nGenerating monthly report..."
auto-browser easy -v -r --site workday "https://workday.com" "Go to timesheet reports, generate report for current month showing hours by project, download as PDF"
