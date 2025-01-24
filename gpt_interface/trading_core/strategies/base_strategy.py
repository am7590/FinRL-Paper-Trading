from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseStrategy(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def generate_signal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal based on current state"""
        pass
    
    @abstractmethod
    async def evaluate_performance(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate strategy performance"""
        pass

class SimpleMomentumStrategy(BaseStrategy):
    async def generate_signal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Simple momentum strategy"""
        market_data = state['market_data']
        portfolio = state['portfolio_value']
        
        # Simple momentum logic (example)
        action = {
            'amount': 0  # Hold by default
        }
        
        # Add your strategy logic here
        return action
    
    async def evaluate_performance(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate strategy performance"""
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        } 