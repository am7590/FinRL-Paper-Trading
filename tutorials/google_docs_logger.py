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
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)

class GoogleDocsLogger:
    def __init__(self, doc_id=None, credentials_path=None):
        """
        Initialize the Google Docs Logger.
        
        Args:
            doc_id (str): The ID of the Google Doc to write to. If None, a new doc will be created.
            credentials_path (str): Path to the Google API credentials file.
        """
        logging.info("Initializing GoogleDocsLogger")
        self.doc_id = doc_id
        self.credentials_path = credentials_path or os.getenv('GOOGLE_DOCS_CREDENTIALS_PATH', 'credentials.json')
        self.token_path = os.path.join(os.path.dirname(self.credentials_path), 'token.pickle')
        
        logging.debug(f"Using credentials path: {self.credentials_path}")
        logging.debug(f"Using token path: {self.token_path}")
        
        # Ensure credentials directory exists
        os.makedirs(os.path.dirname(self.credentials_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.token_path), exist_ok=True)
        
        try:
            self.service = self._get_service()
            self.drive_service = self._get_drive_service()  # Initialize Drive service
            if not doc_id:
                logging.info("No doc_id provided, creating new document")
                self.doc_id = self._create_doc()
                self._initialize_doc()
                self._make_doc_public()  # Make the document publicly viewable
                logging.info(f"Created new document with ID: {self.doc_id}")
                # Print the document URL to terminal
                doc_url = self.get_doc_url()
                print("\n" + "="*80)
                print(f"Logging to Google Doc: {doc_url}")
                print("="*80 + "\n")
            else:
                # Print the document URL to terminal for existing document
                doc_url = self.get_doc_url()
                print("\n" + "="*80)
                print(f"Logging to existing Google Doc: {doc_url}")
                print("="*80 + "\n")
        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            raise
    
    def _get_drive_service(self):
        """Get the Google Drive service with proper authentication."""
        logging.info("Getting Google Drive service")
        SCOPES = ['https://www.googleapis.com/auth/drive']
        creds = None
        
        # Check if we're running in Docker
        is_docker = os.path.exists('/.dockerenv')
        
        # Try service account authentication first if in Docker
        if is_docker:
            try:
                if os.path.exists(self.credentials_path):
                    creds = service_account.Credentials.from_service_account_file(
                        self.credentials_path, scopes=SCOPES)
                    return build('drive', 'v3', credentials=creds)
            except Exception as e:
                logging.error(f"Service account authentication failed for Drive: {str(e)}")
        
        # Fall back to OAuth2 flow for non-Docker environments
        if os.path.exists(self.token_path):
            try:
                with open(self.token_path, 'rb') as token:
                    creds = pickle.load(token)
            except Exception as e:
                logging.error(f"Error loading token for Drive: {str(e)}")
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logging.error(f"Error refreshing token for Drive: {str(e)}")
                    creds = None
            
            if not creds:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(f"Credentials file not found at {self.credentials_path}")
                
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                    
                    with open(self.token_path, 'wb') as token:
                        pickle.dump(creds, token)
                except Exception as e:
                    logging.error(f"Error during OAuth2 flow for Drive: {str(e)}")
                    raise
        
        return build('drive', 'v3', credentials=creds)
    
    def _make_doc_public(self):
        """Make the document publicly viewable."""
        try:
            logging.info("Making document publicly viewable")
            # Create a permission for anyone to view
            permission = {
                'type': 'anyone',
                'role': 'reader'
            }
            
            # Add the permission to the document
            self.drive_service.permissions().create(
                fileId=self.doc_id,
                body=permission,
                fields='id'
            ).execute()
            
            logging.info("Document is now publicly viewable")
        except Exception as e:
            logging.error(f"Error making document public: {str(e)}")
            raise
    
    def _get_service(self):
        """Get the Google Docs service with proper authentication."""
        logging.info("Getting Google Docs service")
        SCOPES = ['https://www.googleapis.com/auth/documents']
        creds = None
        
        # Check if we're running in Docker
        is_docker = os.path.exists('/.dockerenv')
        logging.debug(f"Running in Docker: {is_docker}")
        
        # Try service account authentication first if in Docker
        if is_docker:
            try:
                if os.path.exists(self.credentials_path):
                    logging.info("Attempting service account authentication")
                    with open(self.credentials_path, 'r') as f:
                        logging.debug(f"Credentials file contents: {f.read()[:100]}...")  # Log first 100 chars
                    creds = service_account.Credentials.from_service_account_file(
                        self.credentials_path, scopes=SCOPES)
                    logging.info("Service account authentication successful")
                    return build('docs', 'v1', credentials=creds)
                else:
                    logging.error(f"Credentials file not found at {self.credentials_path}")
            except Exception as e:
                logging.error(f"Service account authentication failed: {str(e)}")
        
        # Fall back to OAuth2 flow for non-Docker environments
        if os.path.exists(self.token_path):
            try:
                logging.info("Attempting to load existing token")
                with open(self.token_path, 'rb') as token:
                    creds = pickle.load(token)
                logging.info("Successfully loaded existing token")
            except Exception as e:
                logging.error(f"Error loading token: {str(e)}")
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    logging.info("Refreshing expired token")
                    creds.refresh(Request())
                    logging.info("Token refreshed successfully")
                except Exception as e:
                    logging.error(f"Error refreshing token: {str(e)}")
                    creds = None
            
            if not creds:
                if not os.path.exists(self.credentials_path):
                    error_msg = f"Credentials file not found at {self.credentials_path}"
                    logging.error(error_msg)
                    raise FileNotFoundError(error_msg)
                
                try:
                    logging.info("Starting OAuth2 flow")
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                    logging.info("OAuth2 flow completed successfully")
                    
                    # Save credentials for future use
                    with open(self.token_path, 'wb') as token:
                        pickle.dump(creds, token)
                    logging.info("Saved credentials to token file")
                except Exception as e:
                    logging.error(f"Error during OAuth2 flow: {str(e)}")
                    raise
        
        return build('docs', 'v1', credentials=creds)
    
    def _create_doc(self):
        """Create a new Google Doc and return its ID."""
        logging.info("Creating new Google Doc")
        title = f'Paper Trading Log - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        try:
            doc = self.service.documents().create(body={'title': title}).execute()
            doc_id = doc.get('documentId')
            logging.info(f"Created new document with ID: {doc_id}")
            return doc_id
        except Exception as e:
            logging.error(f"Error creating document: {str(e)}")
            raise
    
    def _initialize_doc(self):
        """Initialize the document with headers and formatting."""
        logging.info("Initializing document formatting")
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
            logging.info("Document initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing document: {str(e)}")
            raise
    
    def log(self, message, level='INFO'):
        """
        Append a message to the Google Doc with timestamp and log level.
        
        Args:
            message (str): The message to log.
            level (str): Log level (INFO, WARNING, ERROR, TRADE)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.debug(f"Attempting to log message: [{level}] {message}")
        
        try:
            # Get the current document
            doc = self.service.documents().get(documentId=self.doc_id).execute()
            
            # Get the end of the document
            end_index = doc['body']['content'][-1]['endIndex']
            
            # Format the log message
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
            logging.debug("Message logged successfully")
        except Exception as e:
            logging.error(f"Error logging message: {str(e)}")
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
