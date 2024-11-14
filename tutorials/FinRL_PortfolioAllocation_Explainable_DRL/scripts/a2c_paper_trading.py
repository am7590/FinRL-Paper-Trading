import json
from stable_baselines3 import A2C
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.config import INDICATORS

# Load configuration
with open('tutorials/FinRL_PortfolioAllocation_Explainable_DRL/config.json', 'r') as f:
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
a2c_model_path = "tutorials/FinRL_PortfolioAllocation_Explainable_DRL/models/trained_a2c.zip"
a2c_model = A2C.load(a2c_model_path)
print("A2C model loaded successfully!")

paper_trading_a2c = PaperTradingAlpaca(
    ticker_list=ticker_list,
    time_interval=time_interval,
    drl_lib="stable_baselines3",
    agent="a2c",
    cwd=a2c_model_path,
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
