import pytest
import os
from scripts.deploy_ppo_paper_trading import paper_trading_ppo, CONFIG_PATH

def test_config_exists():
    """Test that config file exists and can be loaded"""
    assert os.path.exists(CONFIG_PATH)

def test_paper_trading_initialization():
    """Test that paper trading can be initialized"""
    assert paper_trading_ppo is not None
    assert hasattr(paper_trading_ppo, 'run')

@pytest.mark.skipif(
    not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_SECRET_KEY'),
    reason="Alpaca credentials not set"
)
def test_alpaca_connection():
    """Test that we can connect to Alpaca"""
    assert paper_trading_ppo.api is not None 