from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ..core.agent_manager import FinancialAgentManager
from ..simulation import SimulationEnvironment
from typing import Dict, Any
import logging
import os
from ..trading_core.finrl_wrapper.engine import FinRLEngine, TradingMode, ModelType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FinRL GPT Interface")
agent_manager = FinancialAgentManager()

# Initialize simulation environment
sim_env = SimulationEnvironment()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            message = await websocket.receive_text()
            response = await agent_manager.process_message(message)
            await websocket.send_json(response)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

@app.post("/analyze")
async def analyze_market(request: Dict[str, Any]):
    try:
        analysis = await agent_manager.agents["analyst"].analyze_market(
            request["symbols"],
            request["timeframe"]
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Simulation endpoints
@app.post("/simulation/create")
async def create_simulation(config: Dict[str, Any]):
    """Create a new simulation environment with specified parameters"""
    try:
        sim_id = await sim_env.create_simulation(config)
        return {"simulation_id": sim_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/simulation/{sim_id}/step")
async def simulation_step(sim_id: str, action: Dict[str, Any]):
    """Execute one step in the simulation"""
    try:
        result = await sim_env.step(sim_id, action)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/simulation/{sim_id}/state")
async def get_simulation_state(sim_id: str):
    """Get current state of the simulation"""
    try:
        state = await sim_env.get_state(sim_id)
        return state
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/simulation/backtest")
async def run_backtest(config: Dict[str, Any]):
    """Run a backtest with specified configuration"""
    try:
        results = await sim_env.run_backtest(config)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 

@app.post("/deploy")
async def deploy_model(
    ticker: str,
    model_type: str = "ppo",
    mode: str = "paper"
):
    config = {
        'mode': mode,
        'model_type': model_type,
        'ticker_list': [ticker],
        'technical_indicators': ['macd'],
        'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY'),
        'ALPACA_SECRET_KEY': os.getenv('ALPACA_SECRET_KEY')
    }
    
    engine = FinRLEngine(config)
    await engine.setup_environment()
    model = await engine.train_model()
    
    return {"status": "success", "message": "Model deployed to paper trading"}