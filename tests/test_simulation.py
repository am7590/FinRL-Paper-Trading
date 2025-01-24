import pytest
from gpt_interface.simulation.environment import SimulationEnvironment
from gpt_interface.trading_core.strategies.base_strategy import SimpleMomentumStrategy

@pytest.mark.asyncio
async def test_simulation_workflow():
    # Create simulation environment
    sim_env = SimulationEnvironment()
    
    # Configuration
    config = {
        'initial_amount': 100000,
        'ticker_list': ['AAPL'],
        'technical_indicators': ['macd', 'rsi']
    }
    
    # Create simulation
    sim_id = await sim_env.create_simulation(config)
    
    # Create strategy
    strategy = SimpleMomentumStrategy(config)
    
    # Run a few steps
    for _ in range(5):
        state = await sim_env.get_state(sim_id)
        action = await strategy.generate_signal(state)
        result = await sim_env.step(sim_id, action)
        
        assert 'portfolio_value' in result['state']
        assert 'reward' in result 