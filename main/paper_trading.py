from paper_trading_manager import PaperTradingManager

# Define paths to the configuration and model
config_path = "/tutorials/FinRL_PortfolioAllocation_Explainable_DRL/config.json"
model_path = "tutorials/FinRL_PortfolioAllocation_Explainable_DRL/models/trained_a2c.zip"

# Initialize PaperTradingManager
manager = PaperTradingManager(config_path)

# Start paper trading for A2C model
manager.start_paper_trading("A2C", model_path)