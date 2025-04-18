import logging
import os
import json
from datetime import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/documents']

class GoogleDocsHandler(logging.Handler):
    """A logging handler that writes to Google Docs."""
    
    def __init__(self, doc_id=None, credentials_path=None, token_path=None):
        super().__init__()
        self.doc_id = doc_id
        self.credentials_path = credentials_path or 'credentials.json'
        self.token_path = token_path or 'token.json'
        self.service = None
        self.initialize_service()
        
    def initialize_service(self):
        """Initialize the Google Docs service."""
        creds = None
        # The file token.json stores the user's access and refresh tokens
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
            
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())
                
        self.service = build('docs', 'v1', credentials=creds)
        
    def emit(self, record):
        """Emit a record to Google Docs."""
        try:
            if not self.doc_id:
                # Create a new document if no doc_id is provided
                document = self.service.documents().create(
                    body={'title': f'Trading Log {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'}
                ).execute()
                self.doc_id = document.get('documentId')
                print(f"Created new Google Doc with ID: {self.doc_id}")
                
            # Format the log message
            log_entry = self.format(record)
            
            # Insert the log entry at the end of the document
            self.service.documents().batchUpdate(
                documentId=self.doc_id,
                body={
                    'requests': [
                        {
                            'insertText': {
                                'location': {
                                    'index': 1
                                },
                                'text': log_entry + '\n'
                            }
                        }
                    ]
                }
            ).execute()
        except Exception as e:
            print(f"Error writing to Google Docs: {e}")
            
    def close(self):
        """Close the handler."""
        super().close() 