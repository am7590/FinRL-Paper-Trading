from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os.path
import pickle
from datetime import datetime
import json

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
            self._initialize_doc()
    
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
    
    def _initialize_doc(self):
        """Initialize the document with headers and formatting."""
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
    
    def log(self, message, level='INFO'):
        """
        Append a message to the Google Doc with timestamp and log level.
        
        Args:
            message (str): The message to log.
            level (str): Log level (INFO, WARNING, ERROR, TRADE)
        """
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
