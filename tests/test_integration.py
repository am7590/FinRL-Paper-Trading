import pytest
from gpt_interface.trading_core.finrl_wrapper.engine import FinRLEngine, TradingMode, ModelType
from gpt_interface.api.main import app
from fastapi.testclient import TestClient

@pytest.mark.asyncio
async def test_full_trading_pipeline():
    # Test complete pipeline from model training to execution
    config = {
        'mode': TradingMode.SIMULATION,
        'model_type': ModelType.PPO,
        'ticker_list': ['AAPL'],
        'initial_amount': 100000,
        'technical_indicators': ['macd', 'rsi']
    }
    
    # Initialize engine
    engine = FinRLEngine(config)
    
    # Test environment setup
    env = await engine.setup_environment()
    assert env is not None
    
    # Test model training
    model = await engine.train_model(total_timesteps=100)
    assert model is not None
    
    # Test trading execution
    action = {'amount': 0.5}
    result = await engine.execute_trade(action)
    assert result['state'] is not None 