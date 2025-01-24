import autogen
from typing import Dict, Any
import os

class FinancialAgentManager:
    def __init__(self, config_path: str = "OAI_CONFIG_LIST"):
        self.llm_config = {
            "temperature": 0,
            "config_list": autogen.config_list_from_json(
                config_path, 
                filter_dict={"model": ["gpt-4-0125-preview"]}
            )
        }
        
        # Initialize specialized agents
        self.agents = {
            "trader": self._create_trading_agent(),
            "analyst": self._create_analyst_agent(),
            "researcher": self._create_research_agent()
        }
        
        # Create group chat
        self.group_chat = autogen.GroupChat(
            agents=list(self.agents.values()),
            messages=[],
            max_round=20
        )
        
        self.manager = autogen.GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config
        )
    
    def _create_trading_agent(self):
        return autogen.AssistantAgent(
            name="Trading_Expert",
            llm_config=self.llm_config,
            system_message="""You are an expert in executing trades using FinRL. 
            You understand market dynamics and can implement various trading strategies."""
        )
    
    def _create_analyst_agent(self):
        return autogen.AssistantAgent(
            name="Market_Analyst",
            llm_config=self.llm_config,
            system_message="""You are an expert market analyst using FinGPT capabilities.
            You analyze market conditions and provide insights."""
        )
    
    def _create_research_agent(self):
        return autogen.AssistantAgent(
            name="Research_Expert",
            llm_config=self.llm_config,
            system_message="""You are a research expert who analyzes financial data
            and provides comprehensive market research."""
        )

    async def process_message(self, message: str) -> Dict[str, Any]:
        # Process the message through the agent system
        response = await self.manager.run_chat(
            message,
            sender=self.agents["trader"]
        )
        return {"message": response}

    async def execute_trade(self, message: str) -> Dict[str, Any]:
        """Execute trade based on agent message"""
        # Parse trading intent from message
        trading_intent = await self._parse_trading_intent(message)
        
        # Create simulation if needed
        if not hasattr(self, 'current_sim_id'):
            config = {
                'initial_amount': 100000,
                'ticker_list': ['AAPL', 'GOOGL', 'MSFT'],
                'technical_indicators': ['macd', 'rsi', 'cci', 'dx']
            }
            self.current_sim_id = await self.simulation.create_simulation(config)
        
        # Get current state
        state = await self.simulation.get_state(self.current_sim_id)
        
        # Generate trading signal
        strategy = SimpleMomentumStrategy({})
        action = await strategy.generate_signal(state)
        
        # Execute trade
        result = await self.simulation.step(self.current_sim_id, action)
        
        return result 