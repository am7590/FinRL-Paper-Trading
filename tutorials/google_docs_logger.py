from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os.path
import pickle
from datetime import datetime
import json
import sys

class GoogleDocsLogger:
    def __init__(self, doc_id=None, credentials_path=None):
        """
        Initialize the Google Docs Logger.
        
        Args:
            doc_id (str): The ID of the Google Doc to write to. If None, a new doc will be created.
            credentials_path (str): Path to the Google API credentials file.
        """
        self.doc_id = doc_id
        self.credentials_path = credentials_path or os.getenv('GOOGLE_DOCS_CREDENTIALS_PATH', 'credentials.json')
        self.token_path = os.path.join(os.path.dirname(self.credentials_path), 'token.pickle')
        
        # Ensure credentials directory exists
        os.makedirs(os.path.dirname(self.credentials_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.token_path), exist_ok=True)
        
        self.service = self._get_service()
        if not doc_id:
            self.doc_id = self._create_doc()
            self._initialize_doc()
    
    def _get_service(self):
        """Get the Google Docs service with proper authentication."""
        SCOPES = ['https://www.googleapis.com/auth/documents']
        creds = None
        
        # Check if we're running in Docker
        is_docker = os.path.exists('/.dockerenv')
        
        # Try service account authentication first if in Docker
        if is_docker:
            try:
                if os.path.exists(self.credentials_path):
                    creds = service_account.Credentials.from_service_account_file(
                        self.credentials_path, scopes=SCOPES)
                    return build('docs', 'v1', credentials=creds)
            except Exception as e:
                print(f"Service account authentication failed: {e}", file=sys.stderr)
        
        # Fall back to OAuth2 flow for non-Docker environments
        if os.path.exists(self.token_path):
            try:
                with open(self.token_path, 'rb') as token:
                    creds = pickle.load(token)
            except Exception as e:
                print(f"Error loading token: {e}", file=sys.stderr)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing token: {e}", file=sys.stderr)
                    creds = None
            
            if not creds:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"Credentials file not found at {self.credentials_path}. "
                        "Please place your Google API credentials file in the credentials directory."
                    )
                
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                    
                    # Save credentials for future use
                    with open(self.token_path, 'wb') as token:
                        pickle.dump(creds, token)
                except Exception as e:
                    print(f"Error during authentication: {e}", file=sys.stderr)
                    raise
        
        return build('docs', 'v1', credentials=creds)
    
    def _create_doc(self):
        """Create a new Google Doc and return its ID."""
        title = f'Paper Trading Log - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        try:
            doc = self.service.documents().create(body={'title': title}).execute()
            return doc.get('documentId')
        except Exception as e:
            print(f"Error creating document: {e}", file=sys.stderr)
            raise
    
    def _initialize_doc(self):
        """Initialize the document with headers and formatting."""
        try:
            requests = [
                {
                    'insertText': {
                        'location': {'index': 1},
                        'text': 'Paper Trading Log\n\n'
                    }
                },
                {
                    'updateTextStyle': {
                        'range': {'startIndex': 1, 'endIndex': 16},
                        'textStyle': {
                            'bold': True,
                            'fontSize': {'magnitude': 16, 'unit': 'PT'}
                        },
                        'fields': 'bold,fontSize'
                    }
                }
            ]
            self.service.documents().batchUpdate(
                documentId=self.doc_id,
                body={'requests': requests}
            ).execute()
        except Exception as e:
            print(f"Error initializing document: {e}", file=sys.stderr)
            raise
    
    def log(self, message, level='INFO'):
        """
        Append a message to the Google Doc with timestamp and log level.
        
        Args:
            message (str): The message to log.
            level (str): Log level (INFO, WARNING, ERROR, TRADE)
        """
        try:
            # Get the current document
            doc = self.service.documents().get(documentId=self.doc_id).execute()
            
            # Get the end of the document
            end_index = doc['body']['content'][-1]['endIndex']
            
            # Format the log message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"\n{timestamp} [{level}] - {message}"
            
            # Prepare the request to insert text
            requests = [
                {
                    'insertText': {
                        'location': {'index': end_index - 1},
                        'text': formatted_message
                    }
                }
            ]
            
            # Add formatting based on log level
            if level == 'ERROR':
                requests.append({
                    'updateTextStyle': {
                        'range': {
                            'startIndex': end_index - 1,
                            'endIndex': end_index - 1 + len(formatted_message)
                        },
                        'textStyle': {'foregroundColor': {'color': {'rgbColor': {'red': 1, 'green': 0, 'blue': 0}}}},
                        'fields': 'foregroundColor'
                    }
                })
            elif level == 'WARNING':
                requests.append({
                    'updateTextStyle': {
                        'range': {
                            'startIndex': end_index - 1,
                            'endIndex': end_index - 1 + len(formatted_message)
                        },
                        'textStyle': {'foregroundColor': {'color': {'rgbColor': {'red': 1, 'green': 0.5, 'blue': 0}}}},
                        'fields': 'foregroundColor'
                    }
                })
            elif level == 'TRADE':
                requests.append({
                    'updateTextStyle': {
                        'range': {
                            'startIndex': end_index - 1,
                            'endIndex': end_index - 1 + len(formatted_message)
                        },
                        'textStyle': {'foregroundColor': {'color': {'rgbColor': {'red': 0, 'green': 0.5, 'blue': 0}}}},
                        'fields': 'foregroundColor'
                    }
                })
            
            # Execute the request
            self.service.documents().batchUpdate(
                documentId=self.doc_id,
                body={'requests': requests}
            ).execute()
        except Exception as e:
            print(f"Error logging message: {e}", file=sys.stderr)
            # Don't raise the exception to prevent the trading from stopping
            # Just print the message to stderr as a fallback
            print(f"{timestamp} [{level}] - {message}", file=sys.stderr)
    
    def log_trade(self, action_type, symbol, quantity, price=None, status=None):
        """
        Log a trading action with detailed information.
        
        Args:
            action_type (str): Type of trade (BUY/SELL)
            symbol (str): Stock symbol
            quantity (int): Number of shares
            price (float): Price per share (optional)
            status (str): Status of the trade (optional)
        """
        message = f"{action_type} {quantity} shares of {symbol}"
        if price:
            message += f" @ ${price:.2f}"
        if status:
            message += f" - {status}"
        self.log(message, level='TRADE')
    
    def log_portfolio(self, cash, positions, equity):
        """
        Log portfolio status.
        
        Args:
            cash (float): Available cash
            positions (dict): Current positions
            equity (float): Total portfolio equity
        """
        message = f"\nPortfolio Status:\n"
        message += f"Cash: ${cash:.2f}\n"
        message += f"Total Equity: ${equity:.2f}\n"
        message += "Positions:\n"
        for symbol, position in positions.items():
            message += f"- {symbol}: {position['quantity']} shares @ ${position['price']:.2f}\n"
        self.log(message, level='INFO')
    
    def get_doc_url(self):
        """Get the URL of the Google Doc."""
        return f'https://docs.google.com/document/d/{self.doc_id}'
