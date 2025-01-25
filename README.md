# FinRL-Paper-Trading

This repository provides a Dockerized solution for running multiple deep reinforcement learning (DRL) algorithms for paper trading in stock markets using the [FinRL](https://github.com/AI4Finance-Foundation/FinRL) library. It includes multiple tutorials from FinRL, each implementing different DRL algorithms such as A2C, PPO, TD3, and DDPG, which can be run concurrently to simulate various trading strategies. 

These instructions will help you set up and run the project in a Docker container to ensure compatibility and avoid local dependency issues.

### Prerequisites

- **Docker**: Make sure Docker is installed on your system. You can download it [here](https://www.docker.com/get-started).
- **Git**: Ensure Git is installed if you need to clone the repository.

### Installation

1. **Clone the Repository**
- `git clone https://github.com/am7590/FinRL-Paper-Trading.git`
- `cd FinRL-Paper-Trading`

2. **Initialize our forked FinRL Submodule**
- `git submodule init`
- `git submodule update --recursive`
   
3. **Build the Docker Image**

   Run the following command in the root directory of the project to build the Docker image:

   `docker build -t finrl-paper-trading .`

   This will create a Docker image named finrl-paper-trading, encapsulating all dependencies and configuration needed to run the project.

## Usage

   Run the container interactively to execute each paper trading script as needed:

   `docker run -it --rm --name finrl-container finrl-paper-trading bash`

## Execute Paper Trading Scripts
Navigate to each directory and execute scripts as needed:

- FinRL_PortfolioAllocation_Explainable_DRL (A2C and PPO)
   - `cd ../../FinRL_PortfolioAllocation_Explainable_DRL/scripts`
   - `python a2c_paper_trading.py`
   - `python ppo_paper_trading.py`

- FinRL_PortfolioAllocation_NeurIPS_2020 (TD3 and DDPG)
   - `cd ../../FinRL_PortfolioAllocation_NeurIPS_2020/scripts`
   - `python td3_paper_trading.py`
   - `python ddpg_paper_trading.py`

- FinRL_StockTrading_Fundamental (A2C and PPO)
   - `cd ../../FinRL_StockTrading_Fundamental/scripts`
   - `python a2c_paper_trading.py`
   - `python ppo_paper_trading.py`

- FinRL_StockTrading_NerulIPS_2018 (A2C and PPO)
   - `cd ../../FinRL_StockTrading_NerulIPS_2018/scripts`
   - `python a2c_paper_trading.py`
   - `python ppo_paper_trading.py`

Type exit to close the Docker container when you're finished.

## Optional: Using Docker Compose
To further automate and manage concurrent script execution, you can modify the provided docker-compose.yml to define services for each script.

## Configuration
Each tutorial directory contains a config.json file with model-specific configurations (API keys, trading parameters, etc.). Make sure these are set correctly if any adjustments are needed for different environments.

## Future Improvements
This setup can be adapted for API-based use cases, where an external service (e.g., FinRobot) can trigger each script through an HTTP request or similar. Further modularization into a class structure for each trading strategy could improve integration with external services.

## License
This project is licensed under the MIT License.
