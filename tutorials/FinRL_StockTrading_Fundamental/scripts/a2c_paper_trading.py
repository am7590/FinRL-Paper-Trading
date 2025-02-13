import json
import sys
from stable_baselines3 import A2C
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.config import INDICATORS
import os
import numpy as np
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(ROOT_DIR)
from tutorials.utils.observation_wrapper import ObservationReshapeWrapper

CONFIG_PATH = os.path.join(ROOT_DIR, 'tutorials/FinRL_StockTrading_Fundamental/config.json')
MODEL_PATH = os.path.join(ROOT_DIR, 'tutorials/FinRL_StockTrading_Fundamental/models/trained_a2c.zip')

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
wrapped_model = ObservationReshapeWrapper(a2c_model, model_type="fundamental")

class CustomPaperTradingAlpaca(PaperTradingAlpaca):
    def __init__(self, wrapped_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = wrapped_model
        self.model.observation_space = wrapped_model.observation_space
        
    def trade(self):    
        state = self.get_state()
        print("\n=== Trading Cycle Start ===")
        print(f"Current State Shape: {state.shape}")

        # Update current positions
        positions = self.alpaca.list_positions()
        self.stocks = [0] * len(self.stockUniverse)
        for position in positions:
            if position.symbol in self.stockUniverse:
                idx = self.stockUniverse.index(position.symbol)
                self.stocks[idx] = int(float(position.qty))

        print(f"Current positions: {list(zip(self.stockUniverse, self.stocks))}")
        print(f"Current cash: ${self.cash:.2f}")

        # Get model prediction and scale it
        action = self.model.predict(state)[0]
        print(f"Raw model action: {action}")
        
        scaled_action = action * 100
        scaled_action = scaled_action[:len(self.stockUniverse)]
        print(f"Scaled action: {scaled_action}")

        self.stocks_cd += 1
        if self.turbulence_bool == 0:
            min_action = 10
            
            # Check sell conditions
            sell_indices = np.where(scaled_action < -min_action)[0]
            for index in sell_indices:
                sell_num_shares = min(self.stocks[index], -scaled_action[index])
                qty = abs(int(sell_num_shares))
                if qty > 0:
                    ticker = self.stockUniverse[index]
                    print(f"Executing sell order for {qty} shares of {ticker}")
                    respSO = []
                    self.submitOrder(qty, ticker, "sell", respSO)
                    if respSO[0]:  # If order was successful
                        self.stocks[index] -= qty
                        self.stocks_cd[index] = 0
                    print(f"Sell order response: {respSO}")
            
            # Check buy conditions
            buy_indices = np.where(scaled_action > min_action)[0]
            for index in buy_indices:
                if self.stocks_cd[index] > 0:
                    ticker = self.stockUniverse[index]
                    price = self.price[index]
                    qty = abs(int(scaled_action[index]))
                    if qty > 0 and price * qty < self.cash:
                        print(f"Executing buy order for {qty} shares of {ticker}")
                        respSO = []
                        self.submitOrder(qty, ticker, "buy", respSO)
                        if respSO[0]:  # If order was successful
                            self.stocks[index] += qty
                            self.cash = float(self.alpaca.get_account().cash)
                            self.stocks_cd[index] = 0
                        print(f"Buy order response: {respSO}")
        else:
            print("Turbulence detected, skipping trades")
            
        print(f"Updated positions: {list(zip(self.stockUniverse, self.stocks))}")
        print("=== Trading Cycle End ===\n")

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
    turbulence_thresh=500,  # Using higher threshold
    max_stock=config["trading"]["max_stock"]
)

# Run A2C paper trading
paper_trading_a2c.run()
