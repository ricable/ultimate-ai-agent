# Login Task

## Description
Log into a website using credentials from environment variables.

## Steps
1. Navigate to the login page
2. Find the username input field
3. Enter the username from environment
4. Find the password input field
5. Enter the password from environment
6. Click the login button
7. Verify successful login

## Credentials
LOGIN_USERNAME
LOGIN_PASSWORD

## Expected Output
Successfully logged in, verified by presence of user-specific elements or welcome message.

## Selectors
- Username field: input[name='username'], input[type='email']
- Password field: input[name='password'], input[type='password']
- Login button: button[type='submit'], input[type='submit']
- Success indicator: .welcome-message, .user-profile
