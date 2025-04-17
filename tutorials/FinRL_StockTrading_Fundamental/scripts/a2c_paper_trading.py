import json
from stable_baselines3 import A2C
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.config import INDICATORS
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from tutorials.google_docs_logger import GoogleDocsLogger

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
CONFIG_PATH = os.path.join(ROOT_DIR, 'tutorials/FinRL_StockTrading_Fundamental/config.json')
MODEL_PATH = os.path.join(ROOT_DIR, 'tutorials/FinRL_StockTrading_Fundamental/models/trained_a2c.zip')

# Initialize Google Docs Logger
logger = GoogleDocsLogger()
logger.log("Starting A2C Paper Trading Session")

# Load configuration
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
logger.log(f"Loaded configuration from {CONFIG_PATH}")

# API Keys
TRADING_API_KEY = config["alpaca"]["trading_api_key"]
TRADING_API_SECRET = config["alpaca"]["trading_api_secret"]
TRADING_API_BASE_URL = config["alpaca"]["trading_api_base_url"]

# Model and trading parameters
ticker_list = config["training"]["ticker_list"]
time_interval = config["training"]["time_interval"]
action_dim = len(ticker_list)
state_dim = eval(config["training"]["state_dim_formula"].replace("action_dim", str(action_dim)).replace("INDICATORS", "INDICATORS"))

logger.log(f"Trading parameters: Tickers={ticker_list}, Time Interval={time_interval}, Action Dimension={action_dim}, State Dimension={state_dim}")

# A2C Paper Trading
try:
    a2c_model = A2C.load(MODEL_PATH)
    logger.log("A2C model loaded successfully!")
except Exception as e:
    logger.log(f"Error loading A2C model: {str(e)}")
    raise

paper_trading_a2c = PaperTradingAlpaca(
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

# Override the run method to include logging
original_run = paper_trading_a2c.run
def run_with_logging(self):
    logger.log("Starting paper trading session")
    try:
        result = original_run()
        logger.log("Paper trading session completed successfully")
        return result
    except Exception as e:
        logger.log(f"Error during paper trading: {str(e)}")
        raise

paper_trading_a2c.run = run_with_logging.__get__(paper_trading_a2c)

# Run A2C paper trading
paper_trading_a2c.run()

# Log the Google Doc URL
print(f"Trading logs are being written to: {logger.get_doc_url()}")
