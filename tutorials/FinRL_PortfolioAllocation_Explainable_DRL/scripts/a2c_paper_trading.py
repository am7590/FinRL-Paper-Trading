import json
import builtins
from stable_baselines3 import A2C
from finrl.finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.finrl.config import INDICATORS
import os
import sys
import numpy as np
from datetime import datetime
import logging
import pathlib
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(ROOT_DIR)
from tutorials.utils.observation_wrapper import ObservationReshapeWrapper
from tutorials.FinRL_PortfolioAllocation_Explainable_DRL.scripts.google_docs_handler import GoogleDocsHandler

# Define shared mount location for logs
SHARED_LOG_DIR = os.getenv('SHARED_LOG_DIR', os.path.join(str(Path.home()), 'shared_finrl_logs'))
INSTANCE_LOG_DIR = os.getenv('INSTANCE_LOG_DIR', os.path.join(ROOT_DIR, 'tutorials/FinRL_PortfolioAllocation_Explainable_DRL/trading_logs'))

# Create both local and shared log directories
os.makedirs(SHARED_LOG_DIR, exist_ok=True)
os.makedirs(INSTANCE_LOG_DIR, exist_ok=True)

# Create unique log file names with timestamp
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
shared_log_file = os.path.join(SHARED_LOG_DIR, f'trading_log_{current_time}.txt')
instance_log_file = os.path.join(INSTANCE_LOG_DIR, f'trading_log_{current_time}.txt')

# Configure logging to write to both locations and Google Docs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(shared_log_file),
        logging.FileHandler(instance_log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add Google Docs handler if credentials are available
google_docs_credentials = os.getenv('GOOGLE_DOCS_CREDENTIALS_PATH')
if google_docs_credentials and os.path.exists(google_docs_credentials):
    google_docs_handler = GoogleDocsHandler(
        doc_id=os.getenv('GOOGLE_DOC_ID'),
        credentials_path=google_docs_credentials,
        token_path=os.path.join(ROOT_DIR, 'token.json')
    )
    google_docs_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(google_docs_handler)
    print(f"Google Docs logging enabled. Document ID: {google_docs_handler.doc_id or 'Will create new document'}")
else:
    print("Google Docs logging disabled. Set GOOGLE_DOCS_CREDENTIALS_PATH environment variable to enable.")

def log_and_print(message):
    """Helper function to both log and print messages"""
    print(message)
    logger.info(message)

# Log the setup information
log_and_print(f"Logging setup complete. Logs will be saved to:")
log_and_print(f"Shared location: {shared_log_file}")
log_and_print(f"Instance location: {instance_log_file}")

CONFIG_PATH = os.path.join(ROOT_DIR, 'tutorials/FinRL_PortfolioAllocation_Explainable_DRL/config.json')
MODEL_PATH = os.path.join(ROOT_DIR, 'tutorials/FinRL_PortfolioAllocation_Explainable_DRL/models/trained_a2c.zip')

# Load configuration
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# API Keys
TRADING_API_KEY = config["alpaca"]["trading_api_key"]
TRADING_API_SECRET = config["alpaca"]["trading_api_secret"]
TRADING_API_BASE_URL = config["alpaca"]["trading_api_base_url"]

# Model and trading parameters
ticker_list = config["training"]["ticker_list"]
time_interval = config["training"]["time_interval"]
action_dim = len(ticker_list)
state_dim = eval(config["training"]["state_dim_formula"].replace("action_dim", str(action_dim)).replace("INDICATORS", "INDICATORS"))

# A2C Paper Trading
a2c_model = A2C.load(MODEL_PATH)
print("A2C model loaded successfully!")

# Wrap the model with our observation reshaper
wrapped_model = ObservationReshapeWrapper(a2c_model)
class CustomPaperTradingAlpaca(PaperTradingAlpaca):
    def __init__(self, wrapped_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = wrapped_model
        
    def trade(self):    
        state = self.get_state()
        print("\n=== Trading Cycle Start ===")
        print(f"Current State Shape: {state.shape}")
        print(f"Current State Values: {state}")

        # Get model prediction and scale it
        action = self.model.predict(state)[0]
        # Scale actions from [0,1] to [-100,100] range
        scaled_action = (action * 200) - 100
        scaled_action = scaled_action[:len(self.stockUniverse)]  # Only take first 4 values for our stocks
        
        print(f"\nRaw Predicted Action: {action}")
        print(f"Scaled Action: {scaled_action}")
        print(f"Action Shape: {action.shape}")

        # Current portfolio state
        print("\nPortfolio Status:")
        print(f"Cash: ${self.cash:.2f}")
        print(f"Current Stock Holdings: {self.stocks}")
        print(f"Current Stock Prices: {self.price}")
        
        self.stocks_cd += 1
        if self.turbulence_bool == 0:
            min_action = 10  # stock_cd
            
            # Check sell conditions
            sell_indices = np.where(scaled_action < -min_action)[0]
            print(f"\nPotential Sell Opportunities: {sell_indices}")
            for index in sell_indices:
                sell_num_shares = min(self.stocks[index], -scaled_action[index])
                qty = abs(int(sell_num_shares))
                ticker = self.stockUniverse[index]
                print(f"\nSell Analysis for {ticker}:")
                print(f"- Current Holdings: {self.stocks[index]}")
                print(f"- Sell Action Value: {scaled_action[index]}")
                print(f"- Calculated Sell Quantity: {qty}")
                
                if qty > 0:
                    print(f"Executing sell order for {qty} shares of {ticker}")
                    respSO = []
                    self.submitOrder(qty, ticker, "sell", respSO)
                    print(f"Sell order response: {respSO}")
                    self.cash = float(self.alpaca.get_account().cash)
                    self.stocks_cd[index] = 0
                else:
                    print(f"Skipping sell for {ticker} due to zero quantity")

            # Check buy conditions
            buy_indices = np.where(scaled_action > min_action)[0]
            print(f"\nPotential Buy Opportunities: {buy_indices}")
            for index in buy_indices:
                if self.cash < 0:
                    tmp_cash = 0
                else:
                    tmp_cash = self.cash
                    
                buy_num_shares = min(
                    tmp_cash // self.price[index], abs(int(scaled_action[index]))
                )
                qty = abs(int(buy_num_shares))
                ticker = self.stockUniverse[index]
                
                print(f"\nBuy Analysis for {ticker}:")
                print(f"- Available Cash: ${tmp_cash:.2f}")
                print(f"- Stock Price: ${self.price[index]:.2f}")
                print(f"- Buy Action Value: {scaled_action[index]}")
                print(f"- Calculated Buy Quantity: {qty}")
                
                if qty > 0:
                    print(f"Executing buy order for {qty} shares of {ticker}")
                    respSO = []
                    self.submitOrder(qty, ticker, "buy", respSO)
                    print(f"Buy order response: {respSO}")
                    self.cash = float(self.alpaca.get_account().cash)
                    self.stocks_cd[index] = 0
                else:
                    print(f"Skipping buy for {ticker} due to zero quantity or insufficient funds")
        else:
            print("\nTurbulence detected. Skipping trades to avoid risk.")
            
        print("\n=== Trading Cycle End ===\n")

paper_trading_a2c = CustomPaperTradingAlpaca(
    wrapped_model=wrapped_model,
    ticker_list=ticker_list,
    time_interval=time_interval,
    drl_lib="stable_baselines3",
    agent="a2c",
    cwd=MODEL_PATH,
    net_dim=config["training"]["net_dimension"],
    state_dim=state_dim,
    action_dim=action_dim,
    API_KEY=TRADING_API_KEY,
    API_SECRET=TRADING_API_SECRET,
    API_BASE_URL=TRADING_API_BASE_URL,
    tech_indicator_list=INDICATORS,
    turbulence_thresh=config["trading"]["turbulence_thresh"],
    max_stock=config["trading"]["max_stock"]
)

# Run A2C paper trading
paper_trading_a2c.run()