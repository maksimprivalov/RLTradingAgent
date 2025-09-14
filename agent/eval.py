import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from env import TradingEnv

def evaluate_model(model, df, window_size=30):
    env = TradingEnv(df, window_size=window_size)
    obs, info = env.reset()

    agent_equity = [info["equity"]]

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True) 
        obs, reward, terminated, truncated, info = env.step(action)
        agent_equity.append(info["equity"])

    return np.array(agent_equity)

if __name__ == "__main__":
    df_test = pd.read_csv("data/MSFT_test.csv")

    model = PPO.load("ppo_trader")

    agent_equity = evaluate_model(model, df_test, window_size=30)

    start_price = df_test["Close"].iloc[0]
    end_price = df_test["Close"].iloc[-1]
    bh_growth = df_test["Close"] / start_price * agent_equity[0]

    plt.plot(agent_equity, label="RL Agent")
    plt.plot(bh_growth.values, label="Buy & Hold")
    plt.legend()
    plt.title("RL Agent vs Buy&Hold")
    plt.show()
