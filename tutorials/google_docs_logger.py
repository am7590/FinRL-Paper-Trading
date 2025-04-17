from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os.path
import pickle
from datetime import datetime

class GoogleDocsLogger:
    def __init__(self, doc_id=None, credentials_path='credentials.json'):
        """
        Initialize the Google Docs Logger.
        
        Args:
            doc_id (str): The ID of the Google Doc to write to. If None, a new doc will be created.
            credentials_path (str): Path to the Google API credentials file.
        """
        self.doc_id = doc_id
        self.credentials_path = credentials_path
        self.service = self._get_service()
        if not doc_id:
            self.doc_id = self._create_doc()
    
    def _get_service(self):
        """Get the Google Docs service with proper authentication."""
        SCOPES = ['https://www.googleapis.com/auth/documents']
        creds = None
        
        # Load existing credentials if they exist
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        # If credentials don't exist or are invalid, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for future use
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        return build('docs', 'v1', credentials=creds)
    
    def _create_doc(self):
        """Create a new Google Doc and return its ID."""
        title = f'Paper Trading Log - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        doc = self.service.documents().create(body={'title': title}).execute()
        return doc.get('documentId')
    
    def log(self, message):
        """
        Append a message to the Google Doc.
        
        Args:
            message (str): The message to log.
        """
        # Get the current document
        doc = self.service.documents().get(documentId=self.doc_id).execute()
        
        # Get the end of the document
        end_index = doc['body']['content'][-1]['endIndex']
        
        # Prepare the request to insert text
        requests = [
            {
                'insertText': {
                    'location': {
                        'index': end_index - 1
                    },
                    'text': f'\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {message}'
                }
            }
        ]
        
        # Execute the request
        self.service.documents().batchUpdate(
            documentId=self.doc_id,
            body={'requests': requests}
        ).execute()
    
    def get_doc_url(self):
        """Get the URL of the Google Doc."""
        return f'https://docs.google.com/document/d/{self.doc_id}'
