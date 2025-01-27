from paper_trading_manager import PaperTradingManager

# Work in progress!
# TODO: Fix bug with paths

config_path = "/tutorials/FinRL_PortfolioAllocation_Explainable_DRL/config.json"
model_path = "tutorials/FinRL_PortfolioAllocation_Explainable_DRL/models/trained_a2c.zip"

manager = PaperTradingManager(config_path)

manager.start_paper_trading("A2C", model_path)