#!/usr/bin/env python3
"""
Script to get YouTube OAuth tokens through desktop app flow
"""

import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# If modifying these scopes, delete the file token.pickle.
SCOPES = [
    'https://www.googleapis.com/auth/youtube.force-ssl'  # Full access to YouTube account
]

def get_credentials():
    """Gets valid user credentials from storage or initiates OAuth2 flow."""
    creds = None
    
    # Try to load existing credentials
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Load client secrets from your downloaded client_secrets.json file
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secrets.json',
                SCOPES
            )
            
            print("\nStarting OAuth flow...")
            creds = flow.run_local_server(
                port=0,  # Let it pick any available port
                success_message='The authentication flow has completed. You may close this window.'
            )
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    print("\nHere are your tokens:")
    print(f"Access Token: {creds.token}")
    print(f"Refresh Token: {creds.refresh_token}")
    print("\nAdd these to your .env file!")

if __name__ == '__main__':
    # Enable OAuthlib debugging
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    get_credentials()