import json
from stable_baselines3 import DDPG
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.config import INDICATORS
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
CONFIG_PATH = os.path.join(ROOT_DIR, 'tutorials/FinRL_PortfolioAllocation_NeurIPS_2020/config.json')
MODEL_PATH = os.path.join(ROOT_DIR, 'tutorials/FinRL_PortfolioAllocation_NeurIPS_2020/models/trained_ddpg.zip')

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

# DDPG Paper Trading
ddpg_model = DDPG.load(MODEL_PATH)
print("DDPG model loaded successfully!")

paper_trading_ddpg = PaperTradingAlpaca(
    ticker_list=ticker_list,
    time_interval=time_interval,
    drl_lib="stable_baselines3",
    agent="ddpg",
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

# Run DDPG paper trading
paper_trading_ddpg.run()
    