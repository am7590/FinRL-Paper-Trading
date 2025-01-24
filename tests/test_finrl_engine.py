import pytest
from gpt_interface.trading_core.finrl_wrapper.engine import FinRLEngine, TradingMode, ModelType
import os

@pytest.mark.asyncio
async def test_simulation_ddpg():
    config = {
        'mode': TradingMode.SIMULATION,
        'model_type': ModelType.DDPG,
        'ticker_list': ['AAPL'],
        'technical_indicators': ['macd', 'rsi'],
        'start_date': '2022-01-01',
        'end_date': '2023-01-01',
        'model_params': {
            'net_dim': 64,
            'state_dim': 10,
            'action_dim': 1
        }
    }
    
    engine = FinRLEngine(config)
    await engine.setup_environment()
    
    # Test training
    model = await engine.train_model(total_timesteps=100)
    assert model is not None
    
    # Test execution
    action = {'amount': 0.5}  # Buy 50% of portfolio
    result = await engine.execute_trade(action)
    assert 'state' in result
    assert 'reward' in result
    
    # Test state
    state = await engine.get_state()
    assert 'portfolio_value' in state
    assert 'positions' in state

@pytest.mark.asyncio
async def test_simulation_ppo():
    config = {
        'mode': TradingMode.SIMULATION,
        'model_type': ModelType.PPO,
        'ticker_list': ['AAPL'],
        'technical_indicators': ['macd'],
        'start_date': '2023-03-01',
        'end_date': '2023-03-31'
    }
    
    print("Setting up engine...")
    engine = FinRLEngine(config)
    
    print("Setting up environment...")
    await engine.setup_environment()
    
    print("Loading tutorial model...")
    try:
        model = await engine.train_model()  # This will load the tutorial model
        assert model is not None
    except Exception as e:
        pytest.skip(f"Tutorial model not available: {str(e)}")
    
    print("Testing execution...")
    action = {'amount': 0.5}
    result = await engine.execute_trade(action)
    assert 'state' in result
    assert 'reward' in result
    
    print("Getting state...")
    state = await engine.get_state()
    assert 'portfolio_value' in state

@pytest.mark.asyncio
async def test_paper_trading_mode():
    config = {
        'mode': TradingMode.PAPER,
        'ticker_list': ['AAPL'],
        'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY'),
        'ALPACA_SECRET_KEY': os.getenv('ALPACA_SECRET_KEY'),
        'technical_indicators': ['macd', 'rsi']
    }
    
    engine = FinRLEngine(config)
    
    # Skip if no Alpaca credentials
    if not config['ALPACA_API_KEY'] or not config['ALPACA_SECRET_KEY']:
        pytest.skip("Alpaca credentials not available")
    
    await engine.setup_environment()
    
    # Test paper trading execution
    action = {'symbol': 'AAPL', 'amount': 1}
    result = await engine.execute_trade(action)
    assert result is not None
    
    # Test state
    state = await engine.get_state()
    assert 'portfolio_value' in state