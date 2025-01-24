from typing import Dict, Any, Optional, Union
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add local FinRL to path
finrl_path = Path(__file__).parent.parent.parent.parent / "finrl"
sys.path.append(str(finrl_path))

from finrl.agents.elegantrl.models import DRLAgent  # Only use ElegantRL
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.data_processor import DataProcessor

class TradingMode:
    SIMULATION = "simulation"
    PAPER = "paper"

class ModelType:
    # Update to match ElegantRL algorithms
    PPO = "ppo"
    DDPG = "ddpg"
    TD3 = "td3"
    SAC = "sac"
    A2C = "a2c"
    
    @classmethod
    def get_all(cls):
        return [cls.PPO, cls.DDPG, cls.TD3, cls.SAC, cls.A2C]

class FinRLEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = config.get('mode', TradingMode.SIMULATION)
        self.model_type = config.get('model_type', ModelType.PPO).lower()
        self.env = None
        self.model = None
        self.paper_trading = None
        self.data_processor = DataProcessor(
            data_source=config.get('data_source', 'yahoofinance')
        )
        
    async def setup_environment(self, mode: Optional[str] = None):
        """Initialize or update the trading environment"""
        if mode:
            self.mode = mode
            
        if self.mode == TradingMode.SIMULATION:
            return await self._setup_simulation()
        else:
            return await self._setup_paper_trading()
    
    async def _setup_simulation(self):
        """Setup simulation environment"""
        train_data = await self._fetch_data()
        train_data = self._process_features(train_data)
        
        # Calculate required parameters
        stock_dimension = len(train_data['tic'].unique())
        state_space = stock_dimension * (len(self.config.get('technical_indicators', [])) + 2)
        
        # Initialize environment with all required parameters
        self.env = StockTradingEnv(
            df=train_data,
            stock_dim=stock_dimension,
            hmax=100,  # maximum number of shares to trade
            initial_amount=self.config.get('initial_amount', 100000),
            num_stock_shares=[0] * stock_dimension,  # initially no shares
            buy_cost_pct=[self.config.get('buy_cost_pct', 0.001)] * stock_dimension,  # Make this a list
            sell_cost_pct=[self.config.get('sell_cost_pct', 0.001)] * stock_dimension,  # Make this a list
            state_space=state_space,  # number of features + price + shares
            action_space=stock_dimension,  # action for each stock
            tech_indicator_list=self.config.get('technical_indicators', [
                'macd', 'rsi', 'cci', 'dx'
            ]),
            turbulence_threshold=self.config.get('turbulence_threshold', None),
            reward_scaling=self.config.get('reward_scaling', 1e-4)
        )
        return self.env
    
    async def _setup_paper_trading(self):
        """Setup Alpaca paper trading"""
        self.paper_trading = PaperTradingAlpaca(
            ticker_list=self.config['ticker_list'],
            time_interval=self.config.get('time_interval', '1Min'),
            drl_lib=self.config.get('drl_lib', 'stable_baselines3'),
            agent=self.config.get('agent', 'ppo'),
            API_KEY=self.config['ALPACA_API_KEY'],
            API_SECRET=self.config['ALPACA_SECRET_KEY'],
            API_BASE_URL=self.config.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            tech_indicator_list=self.config.get('technical_indicators', ['macd', 'rsi'])
        )
        return self.paper_trading
    
    async def train_model(self, total_timesteps=10000):
        """Load pre-trained model from tutorial scripts"""
        if self.mode != TradingMode.SIMULATION:
            raise ValueError("Training only available in simulation mode")
        
        if self.model_type not in ModelType.get_all():
            raise ValueError(f"Unsupported model type. Choose from: {ModelType.get_all()}")

        if self.env is None:
            await self._setup_simulation()
        
        # Map model types to their actual model paths from tutorial scripts
        model_paths = {
            ModelType.PPO: "tutorials/FinRL_StockTrading_Fundamental/models/trained_ppo.zip",  # From ppo_paper_trading.py
            ModelType.A2C: "tutorials/FinRL_StockTrading_Fundamental/models/trained_a2c.zip",  # From a2c_paper_trading.py
            ModelType.TD3: "tutorials/FinRL_PortfolioAllocation_NeurIPS_2020/models/trained_td3.zip",  # From paper_trading_td3.py
            ModelType.DDPG: "tutorials/FinRL_StockTrading_NerulIPS_2018/models/agent_ddpg.zip"  # From paper_trading_a2c.py
        }
        
        try:
            model_path = model_paths.get(self.model_type)
            if not model_path:
                raise ValueError(f"No pre-trained model path for {self.model_type}")
            
            print(f"Loading pre-trained model from {model_path}...")
            
            # Load appropriate model type
            if self.model_type == ModelType.PPO:
                from stable_baselines3 import PPO
                self.model = PPO.load(model_path)
            elif self.model_type == ModelType.A2C:
                from stable_baselines3 import A2C
                self.model = A2C.load(model_path)
            elif self.model_type == ModelType.DDPG:
                from stable_baselines3 import DDPG
                self.model = DDPG.load(model_path)
            elif self.model_type == ModelType.TD3:
                from stable_baselines3 import TD3
                self.model = TD3.load(model_path)
            
            print(f"Successfully loaded pre-trained model")
            return self.model
            
        except Exception as e:
            print(f"Could not load tutorial model: {str(e)}")
            print("Available tutorial scripts:")
            print("- tutorials/FinRL_StockTrading_Fundamental/scripts/ppo_paper_trading.py")
            print("- tutorials/FinRL_StockTrading_Fundamental/scripts/a2c_paper_trading.py") 
            print("- tutorials/FinRL_PortfolioAllocation_NeurIPS_2020/scripts/paper_trading_td3.py")
            print("- tutorials/FinRL_StockTrading_NerulIPS_2018/scripts/paper_trading.py")
            raise RuntimeError(f"Failed to load tutorial model: {str(e)}")

    def _df_to_arrays(self, data):
        """Convert DataFrame to arrays required by DRLAgent"""
        # Sort by date and tic
        data = data.sort_values(['date', 'tic']).reset_index(drop=True)
        
        # Get unique dates and tickers
        unique_dates = data.date.unique()
        unique_tickers = data.tic.unique()
        
        # Price array contains close prices
        price_array = data.pivot(index='date', columns='tic', values='close').values
        
        # Tech array contains all technical indicators
        tech_list = self.config.get('technical_indicators', [])
        tech_array = np.hstack([
            data.pivot(index='date', columns='tic', values=tech).values 
            for tech in tech_list
        ]) if tech_list else np.array([])
        
        # Turbulence array (can be empty if not using turbulence)
        turbulence_array = np.zeros(len(unique_dates))
        
        return price_array, tech_array, turbulence_array
    
    async def execute_trade(self, action: Union[Dict[str, Any], np.ndarray]) -> Dict[str, Any]:
        """Execute trade in current environment"""
        if self.mode == TradingMode.SIMULATION:
            if isinstance(action, dict):
                action = self._convert_action_to_array(action)
            
            try:
                # Ensure action has correct shape
                if len(action.shape) == 1:
                    action = action.reshape(-1)
                    
                # Execute step in environment
                step_result = self.env.step(action)
                
                # Debug print
                print(f"Step result type: {type(step_result)}")
                print(f"Step result: {step_result}")
                
                # Handle different return types
                if isinstance(step_result, tuple):
                    if len(step_result) == 5:  # Some environments return 5 values
                        state, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    elif len(step_result) == 4:
                        state, reward, done, info = step_result
                    else:
                        raise ValueError(f"Unexpected number of return values: {len(step_result)}")
                else:
                    state = step_result.get('state')
                    reward = step_result.get('reward')
                    done = step_result.get('done')
                    info = step_result.get('info', {})
                    
                return {
                    'state': state,
                    'reward': reward,
                    'done': done,
                    'info': info
                }
                
            except Exception as e:
                print(f"Error executing trade: {str(e)}")
                raise
        else:
            return await self.paper_trading.execute_trade(action)
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state of the trading environment"""
        if self.mode == TradingMode.SIMULATION:
            if not hasattr(self.env, 'state') or self.env.state is None:
                return {
                    'portfolio_value': 0.0,
                    'positions': None,
                    'market_data': self._get_current_market_data()
                }
            
            # Parse state array:
            # state[0] = portfolio value
            # state[1:stock_dim+1] = stock prices
            # state[stock_dim+1:2*stock_dim+1] = shares held
            # state[2*stock_dim+1:] = technical indicators
            state = self.env.state
            stock_dim = len(self.config.get('ticker_list', ['AAPL']))
            
            positions = {}
            for i in range(stock_dim):
                ticker = self.config['ticker_list'][i]
                positions[ticker] = {
                    'price': float(state[i + 1]),
                    'shares': float(state[i + stock_dim + 1])
                }
            
            return {
                'portfolio_value': float(state[0]),
                'positions': positions,
                'market_data': self._get_current_market_data()
            }
        else:
            return await self.paper_trading.get_state()
    
    def _convert_action_to_array(self, action: Dict[str, Any]) -> np.ndarray:
        """Convert action dictionary to numpy array format"""
        # For single stock, action should be array of shape (1,)
        # Amount between 0 and 1 represents percentage of portfolio to invest
        amount = action.get('amount', 0)
        # Convert to array of shape (stock_dimension,)
        stock_dimension = len(self.config.get('ticker_list', ['AAPL']))
        action_array = np.zeros(stock_dimension)
        action_array[0] = amount  # Apply action to first stock
        return action_array
    
    def _get_current_market_data(self) -> Dict[str, Any]:
        """Get current market data for simulation"""
        if self.env is None or not hasattr(self.env, 'data'):
            return {}
            
        current_step = getattr(self.env, 'current_step', 0)
        
        # Get the row at current_step
        current_data = self.env.df.iloc[current_step]
        
        return {
            'prices': {
                'open': float(current_data['open']),
                'high': float(current_data['high']),
                'low': float(current_data['low']),
                'close': float(current_data['close']),
                'volume': float(current_data['volume'])
            },
            'timestamp': pd.Timestamp(current_data['date']).isoformat()
        }

    def save_model(self, path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
    
    def load_model(self, path: str):
        """Load a trained model"""
        if self.env is None:
            raise ValueError("Environment must be setup before loading model")
        self.model = DRLAgent.load(path)
    
    def _process_features(self, data):
        """Process data with technical indicators"""
        print("Initial data columns:", data.columns.tolist())
        
        # Reset index if timestamp is in index
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()
            print("After reset_index:", data.columns.tolist())
        
        # Standardize column names to lowercase
        column_map = {
            'timestamp': 'date',
            'Datetime': 'date',
            'Date': 'date',
            'index': 'date',  # Add this mapping
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj close'
        }
        data = data.rename(columns=str.lower)  # Convert all to lowercase
        print("After lowercase:", data.columns.tolist())
        
        data = data.rename(columns=column_map)  # Apply any specific mappings
        print("After column_map:", data.columns.tolist())
        
        # Add tic column if not present
        if 'tic' not in data.columns:
            data['tic'] = self.config.get('ticker_list', ['AAPL'])[0]
        
        # Ensure date is datetime type
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        else:
            print("WARNING: 'date' column not found in:", data.columns.tolist())
        
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=self.config.get('technical_indicators', [
                'macd', 'rsi', 'cci', 'dx'
            ])
        )
        
        # Sort by date and tic before processing
        print("Before sort:", data.columns.tolist())
        data = data.sort_values(['date', 'tic'], ignore_index=True)
        processed = fe.preprocess_data(data)
        print("After processing:", processed.columns.tolist())
        return processed
    
    async def _fetch_data(self):
        """Fetch training data"""
        dp = self.data_processor
        try:
            data = await dp.download_data(
                start_date=self.config.get('start_date', '2022-01-01'),
                end_date=self.config.get('end_date', '2023-01-01'),
                ticker_list=self.config.get('ticker_list', ['AAPL']),
                time_interval=self.config.get('time_interval', '1d')
            )
            
            # Print the data structure for debugging
            print(f"Downloaded data columns: {data.columns.tolist()}")
            
            # Rename columns immediately after download
            if 'timestamp' in data.columns:
                data = data.rename(columns={'timestamp': 'date'})
            
            # Convert column names to lowercase for consistency
            data.columns = data.columns.str.lower()
            
            # Drop Adj Close if present
            if 'adj close' in data.columns:
                data = data.drop(columns=['adj close'])
            
            return data
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            # Return mock data with correct column names
            dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='B')
            mock_data = pd.DataFrame({
                'date': dates,
                'open': np.random.randn(len(dates)) * 10 + 100,
                'high': np.random.randn(len(dates)) * 10 + 102,
                'low': np.random.randn(len(dates)) * 10 + 98,
                'close': np.random.randn(len(dates)) * 10 + 101,
                'volume': np.random.randint(1000000, 10000000, len(dates)),
                'tic': 'AAPL'
            })
            return mock_data 