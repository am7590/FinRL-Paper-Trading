import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Get the logger
logger = logging.getLogger('finrl_trading')

def setup_logging(shared_log_dir=None, instance_log_dir=None):
    """Set up logging configuration"""
    if not shared_log_dir:
        shared_log_dir = os.getenv('SHARED_LOG_DIR', os.path.join(str(Path.home()), 'shared_finrl_logs'))
    if not instance_log_dir:
        instance_log_dir = os.getenv('INSTANCE_LOG_DIR', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trading_logs'))
    
    # Create directories if they don't exist
    os.makedirs(shared_log_dir, exist_ok=True)
    os.makedirs(instance_log_dir, exist_ok=True)
    
    # Create unique log file names with timestamp
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    shared_log_file = os.path.join(shared_log_dir, f'trading_log_{current_time}.txt')
    instance_log_file = os.path.join(instance_log_dir, f'trading_log_{current_time}.txt')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(shared_log_file),
            logging.FileHandler(instance_log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Try to add Google Docs handler if available
    try:
        from tutorials.FinRL_PortfolioAllocation_Explainable_DRL.scripts.google_docs_handler import GoogleDocsHandler
        google_docs_credentials = os.getenv('GOOGLE_DOCS_CREDENTIALS_PATH')
        if google_docs_credentials and os.path.exists(google_docs_credentials):
            google_docs_handler = GoogleDocsHandler(
                doc_id=os.getenv('GOOGLE_DOC_ID'),
                credentials_path=google_docs_credentials,
                token_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'token.json')
            )
            google_docs_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(google_docs_handler)
            print(f"Google Docs logging enabled. Document ID: {google_docs_handler.doc_id or 'Will create new document'}")
    except ImportError:
        print("Google Docs handler not available")
    
    return logger

def log_and_print(message):
    """Helper function to both log and print messages"""
    print(message)
    logger.info(message)

# Initialize logger if not already initialized
if not logger.handlers:
    setup_logging() 