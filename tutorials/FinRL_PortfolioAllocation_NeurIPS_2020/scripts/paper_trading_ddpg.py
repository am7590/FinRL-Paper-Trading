import json
from stable_baselines3 import DDPG
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.config import INDICATORS
import os
import numpy as np
import threading

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

class ReshapingPaperTradingAlpaca(PaperTradingAlpaca):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store state_dim from kwargs
        self.state_dim = kwargs.get('state_dim')
        if self.state_dim is None:
            raise ValueError("state_dim must be provided")
        # Initialize portfolio value
        self.portfolio_value = 100000  # Starting with $100,000
        self.initial_portfolio_value = self.portfolio_value
        # Set the model
        self.model = ddpg_model  # Use the globally loaded model
        # Override turbulence check
        self.ignore_turbulence = True  # Add this flag to force trading
        print("Initialized ReshapingPaperTradingAlpaca with:")
        print(f"- Portfolio Value: ${self.portfolio_value}")
        print(f"- State Dim: {self.state_dim}")
        print(f"- Stock Universe: {self.stockUniverse}")
        print(f"- Ignore Turbulence: {self.ignore_turbulence}")

    def trade(self):
        print("\n=== Starting Trading Round ===")
        state = self.get_state()
        print(f"Raw state shape: {state.shape}")
        
        # The state has 333 elements:
        # - 1 (initial amount)
        # - 2 (turbulence and turbulence_bool)
        # - 3 * 30 = 90 (price, stocks, stocks_cd for each stock)
        # - 8 * 30 = 240 (technical indicators for each stock)
        
        n_stocks = len(self.stockUniverse)  # 30
        n_indicators = len(self.tech_indicator_list)  # 8
        
        # Extract components
        amount = state[0]
        turbulence = state[1:3]
        stock_features = state[3:3+3*n_stocks].reshape(3, n_stocks)  # price, stocks, stocks_cd
        tech_indicators = state[3+3*n_stocks:].reshape(n_indicators, n_stocks)
        
        # Stack all features to get shape (14, 30)
        amount_row = np.full((1, n_stocks), amount)
        turbulence_rows = np.tile(turbulence, (n_stocks, 1)).T
        state = np.vstack([amount_row, turbulence_rows, stock_features, tech_indicators])
        print(f"Intermediate state shape: {state.shape}")
        
        # Reshape to (14, 6) by taking first 6 stocks
        # This is a temporary solution - we should retrain the model for 30 stocks
        state = state[:, :6]
        print(f"Final state shape: {state.shape}")
        
        # Add batch dimension for model input
        reshaped_state = state.reshape(1, *state.shape)
        
        print("\n=== Getting Model Prediction ===")
        if self.drl_lib == "stable_baselines3":
            action = self.model.predict(reshaped_state)[0]
            print(f"Raw model actions: {action}")
            
            # Since the model only outputs 6 actions, we need to pad with zeros for remaining stocks
            full_action = np.zeros(n_stocks)
            full_action[:6] = action
            action = full_action
            print(f"Padded actions: {action}")
        else:
            raise ValueError("Only stable_baselines3 is supported in this implementation")

        print("\n=== Processing Actions ===")
        self.stocks_cd += 1
        
        # Check if we should proceed with trading
        should_trade = self.ignore_turbulence or self.turbulence_bool == 0
        if should_trade:
            min_action = 10  # stock_cd
            
            # Normalize actions to ensure they sum to 1
            action = np.clip(action, 0, 1)  # Ensure non-negative weights
            action = action / np.sum(action)  # Normalize to sum to 1
            print(f"Normalized actions (portfolio weights): {action}")
            
            # Get account information
            print("\n=== Getting Account Information ===")
            try:
                account = self.alpaca.get_account()
                self.portfolio_value = float(account.equity)
                self.cash = float(account.cash)
                print(f"Current Portfolio Value: ${self.portfolio_value}")
                print(f"Available Cash: ${self.cash}")
                print(f"Buying Power: ${float(account.buying_power)}")
                
                current_stocks = self.stocks.copy()
                print(f"Current stock holdings: {current_stocks}")
                
                print("\n=== Getting Current Stock Prices ===")
                current_prices = []
                for symbol in self.stockUniverse:
                    try:
                        price = float(self.alpaca.get_latest_trade(symbol).price)
                        current_prices.append(price)
                        print(f"{symbol}: ${price:.2f}")
                    except Exception as e:
                        print(f"Error getting price for {symbol}: {str(e)}")
                        current_prices.append(0)  # Use 0 as placeholder for failed price fetches
                
                current_prices = np.array(current_prices)
                
                # Calculate current position values
                current_position_values = current_stocks * current_prices
                total_position_value = np.sum(current_position_values)
                print(f"\nCurrent Position Values: ${current_position_values}")
                print(f"Total Position Value: ${total_position_value}")
                
                # Calculate target shares based on portfolio value and weights
                # Use cash + current positions as the base for calculations
                available_value = self.cash + total_position_value
                target_shares = np.floor(action * available_value / current_prices).astype(int)
                print(f"\nTarget stock quantities: {target_shares}")
                
                # Calculate differences from current holdings
                share_differences = target_shares - current_stocks
                print(f"Share differences: {share_differences}")
                
                # Calculate required cash for buys
                buy_indices = share_differences > 0
                total_buy_cost = np.sum(share_differences[buy_indices] * current_prices[buy_indices])
                print(f"\nTotal cost for buys: ${total_buy_cost:.2f}")
                print(f"Available cash: ${self.cash:.2f}")
                
                if total_buy_cost > self.cash:
                    print(f"\nWARNING: Not enough cash for all buys. Scaling back orders...")
                    # Scale back buy orders to fit available cash
                    scale_factor = self.cash / total_buy_cost
                    share_differences[buy_indices] = np.floor(share_differences[buy_indices] * scale_factor).astype(int)
                    print(f"Scaled share differences: {share_differences}")
                
                print("\n=== Executing Trades ===")
                # Execute trades
                for i, diff in enumerate(share_differences):
                    if abs(diff) >= min_action and current_prices[i] > 0:  # Only trade if difference is significant and we have a valid price
                        side = "buy" if diff > 0 else "sell"
                        qty = abs(int(diff))
                        trade_value = qty * current_prices[i]
                        
                        # For buys, check if we have enough cash
                        if side == "buy" and trade_value > self.cash:
                            print(f"\nSkipping {side} for {self.stockUniverse[i]} - Not enough cash:")
                            print(f"- Required: ${trade_value:.2f}")
                            print(f"- Available: ${self.cash:.2f}")
                            continue
                        
                        print(f"\nExecuting trade for {self.stockUniverse[i]}:")
                        print(f"- Direction: {side}")
                        print(f"- Quantity: {qty}")
                        print(f"- Target Weight: {action[i]:.2%}")
                        print(f"- Current Price: ${current_prices[i]:.2f}")
                        print(f"- Trade Value: ${trade_value:.2f}")
                        
                        # Submit the order
                        try:
                            self.alpaca.submit_order(
                                symbol=self.stockUniverse[i],
                                qty=qty,
                                side=side,
                                type='market',
                                time_in_force='day'
                            )
                            print(f"Order submitted successfully")
                            # Update cash balance
                            if side == "buy":
                                self.cash -= trade_value
                            else:
                                self.cash += trade_value
                        except Exception as e:
                            print(f"Order failed: {str(e)}")
                
                print("\nTrading round completed.")
                
            except Exception as e:
                print(f"Error during trading: {str(e)}")
            
        else:
            print("\nSkipping trades due to high turbulence (override is off)")
            
        # Update portfolio value after trading
        try:
            account = self.alpaca.get_account()
            self.portfolio_value = float(account.equity)
            self.cash = float(account.cash)
            print(f"\nFinal Portfolio Status:")
            print(f"- Portfolio Value: ${self.portfolio_value}")
            print(f"- Cash: ${self.cash}")
            print(f"- Buying Power: ${float(account.buying_power)}")
        except Exception as e:
            print(f"Error getting updated portfolio value: {str(e)}")

paper_trading_ddpg = ReshapingPaperTradingAlpaca(
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
    