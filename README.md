# FinRL-Paper-Trading

This repository provides a Dockerized solution for running multiple deep reinforcement learning (DRL) algorithms for paper trading in stock markets using the [FinRL](https://github.com/AI4Finance-Foundation/FinRL) library. It includes multiple tutorials from FinRL, each implementing different DRL algorithms such as A2C, PPO, TD3, and DDPG, which can be run concurrently to simulate various trading strategies. 

These instructions will help you set up and run the project in a Docker container to ensure compatibility and avoid local dependency issues.

### Prerequisites

- **Docker**: Download it [here](https://www.docker.com/get-started).

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

## Usage

   `docker run -it --rm --name finrl-container finrl-paper-trading bash`

## Important Note About Model Input Shape
Ensuring that your input data matches the model's expected shape is crucial to prevent shape mismatch errors in financial reinforcement learning models. These errors often arise from discrepancies in the number of tickers (assets), technical indicators, or the data window length used during preprocessing.​

### Key Factors Influencing Input Shape:
- Number of Tickers (Assets): The total number of financial assets included in your dataset.​
- Number of Technical Indicators: The set of calculated metrics (e.g., moving averages, RSI) applied to each asset.​
- Data Window Length: The number of historical time steps considered for each observation.​

### Typical Expected Input Shape:
Models often expect input data in a three-dimensional array with the shape (number of tickers, number of indicators, window length). For example, with 20 tickers, 16 indicators, and a window length of 10, the expected input shape would be (20, 16, 10).​

### Recommendations to Resolve Shape Mismatch Errors:
- Double check the paper trading script's config file for timeframe, number of indicators, and number of stock tickers. Reference this to the saved colab files in the shared drive, where the configs are saved. Make sure (number of tickers, number of indicators, window length) matches up between the two configs.
- Verify Data Dimensions: Ensure that your data maintains the correct multi-dimensional structure throughout the preprocessing pipeline.​
- Consistent Preprocessing: Apply the same preprocessing steps to both training and paper trading scripts. The colab notebooks used to train all models in this repo are available in our shared drive for reference.

## Execute Paper Trading Scripts
Navigate to each directory and execute scripts as needed:

### Available Scripts
- FinRL_PortfolioAllocation_Explainable_DRL (A2C and PPO)
   ```bash
   cd tutorials/FinRL_PortfolioAllocation_Explainable_DRL/scripts
   python a2c_paper_trading.py
   python ppo_paper_trading.py
   ```

- FinRL_PortfolioAllocation_NeurIPS_2020 (TD3 and DDPG)
   ```bash
   cd tutorials/FinRL_PortfolioAllocation_NeurIPS_2020/scripts
   python td3_paper_trading.py
   python ddpg_paper_trading.py
   ```

- FinRL_StockTrading_Fundamental (A2C and PPO)
   ```bash
   cd tutorials/FinRL_StockTrading_Fundamental/scripts
   python a2c_paper_trading.py
   python ppo_paper_trading.py
   ```

- FinRL_StockTrading_NerulIPS_2018 (A2C and PPO)
   ```bash
   cd tutorials/FinRL_StockTrading_NerulIPS_2018/scripts
   python a2c_paper_trading.py
   python ppo_paper_trading.py
   ```

# Delpoying to AWS EC2

## 1. Setup EC2 environment

- Update the package index:

  ```sudo yum update -y```

- Install Git:
  
   ```sudo yum install -y git```

- Install Docker:
  
   ```sudo yum install -y docker```

- Start Docker service:

   ```sudo systemctl start docker```

- Enable Docker to start on boot:
  
   ```sudo systemctl enable docker```

- Verify Docker installation:

   ```docker --version```

## 2. Run a paper trading script in the background

- Install screen if not already installed:
  
   ```sudo yum install screen -y```

- NOTE: Screens have a 1 to 1 relationship with paper trading scripts.
- ANOTHER NOTE: Don't use 'finrobot' for your own images! Be more specific.

- Create a new screen session:

  ```screen -S finrobot```

- Inside the screen session, run your docker compose command:

  ```sudo docker run -it --rm --name finrl-container finrl-paper-trading bash```

## 3. Final steps
- Detach from screen session (don't close it) by pressing: Ctrl+A then D
- To reattach to the screen session later:

  ```screen -r finrobot```



## License
This project is licensed under the MIT License.
