<<<<<<< HEAD
import sys
import os

# Add the project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(ROOT_DIR)

# Rest of your imports...
from finrl.finrl.meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading as PaperTradingAlpaca

=======
<<<<<<< Updated upstream
>>>>>>> e863c9d1ff54bdae2107f7e299a665beaedb58af
import json
from stable_baselines3 import A2C
# from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.finrl.config import INDICATORS
import os
from tutorials.utils.observation_wrapper import ObservationReshapeWrapper

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
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

# Add error handling for model loading
try:
    a2c_model = A2C.load(MODEL_PATH)
    print("A2C model loaded successfully!")
except Exception as e:
    print(f"Error loading A2C model: {e}")
    raise

# Wrap the model with our observation reshaper
wrapped_model = ObservationReshapeWrapper(a2c_model)

try:
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
        max_stock=config["trading"]["max_stock"],
        model=wrapped_model
    )

<<<<<<< HEAD
    # Run A2C paper trading
    paper_trading_a2c.run()
except Exception as e:
    print(f"Error during paper trading: {e}")
    raise
=======
# Run A2C paper trading
paper_trading_a2c.run()
=======
import json
from stable_baselines3 import A2C
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.config import INDICATORS
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
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

# Run A2C paper trading
paper_trading_a2c.run()
>>>>>>> Stashed changes
>>>>>>> e863c9d1ff54bdae2107f7e299a665beaedb58af
