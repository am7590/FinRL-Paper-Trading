import json
import os
from stable_baselines3 import A2C, PPO, DDPG, TD3
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.config import INDICATORS

class PaperTradingManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_env()

    def load_config(self):
        """Load configuration from JSON file."""
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def setup_env(self):
        """Set environment variables and API keys based on config."""
        self.DATA_API_KEY = self.config["alpaca"]["data_api_key"]
        self.DATA_API_SECRET = self.config["alpaca"]["data_api_secret"]
        self.DATA_API_BASE_URL = self.config["alpaca"]["data_api_base_url"]
        self.TRADING_API_KEY = self.config["alpaca"]["trading_api_key"]
        self.TRADING_API_SECRET = self.config["alpaca"]["trading_api_secret"]
        self.TRADING_API_BASE_URL = self.config["alpaca"]["trading_api_base_url"]
        self.ticker_list = self.config["training"]["ticker_list"]
        self.time_interval = self.config["training"]["time_interval"]
        self.net_dimension = self.config["training"]["net_dimension"]
        self.state_dim = eval(self.config["training"]["state_dim_formula"].replace("action_dim", str(len(self.ticker_list))).replace("INDICATORS", "INDICATORS"))
        self.action_dim = len(self.ticker_list)

    def load_model(self, model_name, model_path):
        """Load specified model."""
        if model_name == "A2C":
            return A2C.load(model_path)
        elif model_name == "PPO":
            return PPO.load(model_path)
        elif model_name == "DDPG":
            return DDPG.load(model_path)
        elif model_name == "TD3":
            return TD3.load(model_path)
        else:
            raise ValueError(f"Model {model_name} is not supported.")

    def start_paper_trading(self, model_name, model_path):
        """Set up and run paper trading with the specified model."""
        model = self.load_model(model_name, model_path)
        print(f"{model_name} model loaded successfully!")

        paper_trading = PaperTradingAlpaca(
            ticker_list=self.ticker_list,
            time_interval=self.time_interval,
            drl_lib="stable_baselines3",
            agent=model_name.lower(),
            cwd=model_path,
            net_dim=self.net_dimension,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            API_KEY=self.TRADING_API_KEY,
            API_SECRET=self.TRADING_API_SECRET,
            API_BASE_URL=self.TRADING_API_BASE_URL,
            tech_indicator_list=INDICATORS,
            turbulence_thresh=self.config["trading"]["turbulence_thresh"],
            max_stock=self.config["trading"]["max_stock"]
        )

        paper_trading.run()
