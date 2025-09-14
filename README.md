# RL TradingAgent
### A reinforcement learning agent which traids on the stock market.

## Overview
This project implements a reinforcement learning (RL) agent that learns how to trade a stock (e.g., MSFT) using daily price data. 
The agent is trained with PPO (Proximal Policy Optimization) from the `stable-baselines3` library and operates in a custom 
Gymnasium environment.

## Project Structure
- **features.py**  
  Responsible for downloading and preparing financial data using `yfinance`.  
  Features currently included:
  - Log returns
  - Simple moving averages (SMA20, SMA50)
  - Volatility (20-day rolling std of log returns)
  - Exponential moving averages (EMA12, EMA26)
  - MACD & MACD signal line
  - RSI14
  - Bollinger Bands (mid, up, down)
  - Volume change (%)

- **prepare_data.py**  
  Splits data into train, test, and validation sets. Saves them as CSV for easy reuse.

- **env.py**  
  Custom trading environment:
  - State (observation): last N=30 days of selected features, flattened into a vector.
  - Action space: {0 = SELL, 1 = HOLD, 2 = BUY}
  - Reward: based on `log_return × position` (agent earns daily return if in market).
  - Commission/fee penalty applied on trades.
  - Small penalty on HOLD to discourage always staying invested.

- **train.py**  
  Loads training data and trains PPO agent using `stable-baselines3`.  
  Saves trained model as `ppo_trader.zip`.

- **eval.py**  
  Loads trained model and evaluates it on test data.  
  Compares RL agent's equity curve with Buy&Hold baseline.  
  Plots performance curves and calculates risk/return metrics.

## Current Results
- The RL agent reproduces Buy&Hold behavior in simple cases but is beginning to learn when to exit the market.  
- With extended features (RSI, MACD, volatility, etc.), the agent shows smoother equity curves (less drawdown).  
- Buy&Hold still outperforms in absolute profit, but RL shows potential in risk management.

## Next Steps
- Improve reward shaping to balance risk and profit.
- Train for longer (500k+ timesteps).
- Add more advanced baselines (e.g., SMA crossover) for comparison.
- Optimize hyperparameters (learning rate, policy network size).
