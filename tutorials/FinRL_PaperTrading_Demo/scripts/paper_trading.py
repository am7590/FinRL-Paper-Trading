import json
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.meta.paper_trading.common import train, test, alpaca_history, DIA_history
from finrl.config import INDICATORS
from finrl.config_tickers import DOW_30_TICKER
import datetime
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
from matplotlib import ticker
import gym

# Load config from JSON file
with open('../config.json', 'r') as f:
    config = json.load(f)

# API Keys
DATA_API_KEY = config["alpaca"]["data_api_key"]
DATA_API_SECRET = config["alpaca"]["data_api_secret"]
DATA_API_BASE_URL = config["alpaca"]["data_api_base_url"]
TRADING_API_KEY = config["alpaca"]["trading_api_key"]
TRADING_API_SECRET = config["alpaca"]["trading_api_secret"]
TRADING_API_BASE_URL = config["alpaca"]["trading_api_base_url"]

# Model Parameters
ticker_list = config["training"]["ticker_list"]
env = StockTradingEnv

# Updated ERL_PARAMS based on config values
ERL_PARAMS = {
    "learning_rate": 3e-6,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": config["training"]["net_dimension"],
    "target_step": config["training"]["target_step"],
    "eval_gap": config["training"]["eval_gap"],
    "eval_times": config["training"]["eval_times"]
}

# Set up date ranges from config
train_start_date = config["dates"]["train_start_date"]
train_end_date = config["dates"]["train_end_date"]
test_start_date = config["dates"]["test_start_date"]
test_end_date = config["dates"]["test_end_date"]
full_train_start_date = config["dates"]["full_train_start_date"]
full_train_end_date = config["dates"]["full_train_end_date"]

# Calculate state_dim based on formula in config
action_dim = len(ticker_list)
state_dim = eval(config["training"]["state_dim_formula"].replace("action_dim", str(action_dim)).replace("INDICATORS", "INDICATORS"))

# Set up Paper Trading
paper_trading_erl = PaperTradingAlpaca(
    ticker_list=ticker_list,
    time_interval=config["training"]["time_interval"],
    drl_lib="elegantrl",
    agent="ppo",
    cwd="../papertrading_erl_retrain",
    net_dim=ERL_PARAMS["net_dimension"],
    state_dim=state_dim,
    action_dim=action_dim,
    API_KEY=TRADING_API_KEY,
    API_SECRET=TRADING_API_SECRET,
    API_BASE_URL=TRADING_API_BASE_URL,
    tech_indicator_list=INDICATORS,
    turbulence_thresh=config["trading"]["turbulence_thresh"],
    max_stock=config["trading"]["max_stock"]
)

# Run the paper trading simulation
paper_trading_erl.run()
