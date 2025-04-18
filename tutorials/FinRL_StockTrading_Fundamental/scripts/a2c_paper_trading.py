import json
import sys
from stable_baselines3 import A2C
from finrl.finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.finrl.config import INDICATORS
import os
import numpy as np
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(ROOT_DIR)
from tutorials.utils.observation_wrapper import ObservationReshapeWrapper
from tutorials.google_docs_logger import GoogleDocsLogger

CONFIG_PATH = os.path.join(ROOT_DIR, 'tutorials/FinRL_StockTrading_Fundamental/config.json')
MODEL_PATH = os.path.join(ROOT_DIR, 'tutorials/FinRL_StockTrading_Fundamental/models/trained_a2c.zip')

# Initialize Google Docs Logger
logger = GoogleDocsLogger()
logger.log("Starting A2C Paper Trading Session", level='INFO')

# Load configuration
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
logger.log(f"Loaded configuration from {CONFIG_PATH}", level='INFO')

# API Keys
TRADING_API_KEY = config["alpaca"]["trading_api_key"]
TRADING_API_SECRET = config["alpaca"]["trading_api_secret"]
TRADING_API_BASE_URL = config["alpaca"]["trading_api_base_url"]

# Model and trading parameters
ticker_list = config["training"]["ticker_list"]
time_interval = config["training"]["time_interval"]
action_dim = len(ticker_list)
state_dim = eval(config["training"]["state_dim_formula"].replace("action_dim", str(action_dim)).replace("INDICATORS", "INDICATORS"))

logger.log(f"Trading parameters: Tickers={ticker_list}, Time Interval={time_interval}, Action Dimension={action_dim}, State Dimension={state_dim}", level='INFO')

# A2C Paper Trading
try:
    a2c_model = A2C.load(MODEL_PATH)
    logger.log("A2C model loaded successfully!", level='INFO')
except Exception as e:
    logger.log(f"Error loading A2C model: {str(e)}", level='ERROR')
    raise

# Wrap the model with our observation reshaper
wrapped_model = ObservationReshapeWrapper(a2c_model, model_type="fundamental")

class CustomPaperTradingAlpaca(PaperTradingAlpaca):
    def __init__(self, wrapped_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = wrapped_model
        self.model.observation_space = wrapped_model.observation_space
        
    def trade(self):    
        state = self.get_state()
        logger.log("\n=== Trading Cycle Start ===", level='INFO')
        logger.log(f"Current State Shape: {state.shape}", level='INFO')

        # Update current positions
        positions = self.alpaca.list_positions()
        self.stocks = [0] * len(self.stockUniverse)
        for position in positions:
            if position.symbol in self.stockUniverse:
                idx = self.stockUniverse.index(position.symbol)
                self.stocks[idx] = int(float(position.qty))

        # Log current portfolio status
        portfolio = {pos.symbol: {'quantity': float(pos.qty), 'price': float(pos.current_price)} for pos in positions}
        logger.log_portfolio(
            cash=float(self.alpaca.get_account().cash),
            positions=portfolio,
            equity=float(self.alpaca.get_account().equity)
        )

        # Get model prediction and scale it
        action = self.model.predict(state)[0]
        logger.log(f"Raw model action: {action}", level='INFO')
        
        scaled_action = action * 100
        scaled_action = scaled_action[:len(self.stockUniverse)]
        logger.log(f"Scaled action: {scaled_action}", level='INFO')

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
                    logger.log_trade(
                        action_type='SELL',
                        symbol=ticker,
                        quantity=qty,
                        price=self.price[index],
                        status='PENDING'
                    )
                    respSO = []
                    self.submitOrder(qty, ticker, "sell", respSO)
                    if respSO[0]:  # If order was successful
                        self.stocks[index] -= qty
                        self.stocks_cd[index] = 0
                        logger.log_trade(
                            action_type='SELL',
                            symbol=ticker,
                            quantity=qty,
                            price=self.price[index],
                            status='COMPLETED'
                        )
            
            # Check buy conditions
            buy_indices = np.where(scaled_action > min_action)[0]
            for index in buy_indices:
                if self.stocks_cd[index] > 0:
                    ticker = self.stockUniverse[index]
                    price = self.price[index]
                    qty = abs(int(scaled_action[index]))
                    if qty > 0 and price * qty < self.cash:
                        logger.log_trade(
                            action_type='BUY',
                            symbol=ticker,
                            quantity=qty,
                            price=price,
                            status='PENDING'
                        )
                        respSO = []
                        self.submitOrder(qty, ticker, "buy", respSO)
                        if respSO[0]:  # If order was successful
                            self.stocks[index] += qty
                            self.cash = float(self.alpaca.get_account().cash)
                            self.stocks_cd[index] = 0
                            logger.log_trade(
                                action_type='BUY',
                                symbol=ticker,
                                quantity=qty,
                                price=price,
                                status='COMPLETED'
                            )
        else:
            logger.log("Turbulence detected, skipping trades", level='WARNING')
            
        # Log updated portfolio status
        positions = self.alpaca.list_positions()
        portfolio = {pos.symbol: {'quantity': float(pos.qty), 'price': float(pos.current_price)} for pos in positions}
        logger.log_portfolio(
            cash=float(self.alpaca.get_account().cash),
            positions=portfolio,
            equity=float(self.alpaca.get_account().equity)
        )
        
        logger.log("=== Trading Cycle End ===\n", level='INFO')

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
try:
    paper_trading_a2c.run()
    logger.log("Paper trading session completed successfully", level='INFO')
except Exception as e:
    logger.log(f"Error during paper trading: {str(e)}", level='ERROR')
    raise
finally:
    # Log the Google Doc URL
    print(f"Trading logs are being written to: {logger.get_doc_url()}")
