from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
from env import TradingEnv

if __name__ == "__main__":
    df = pd.read_csv("data/MSFT_train.csv")

    env = DummyVecEnv([lambda: TradingEnv(df, window_size=30)])

    model = PPO("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=100_000)

    model.save("ppo_trader")

    print("Learning completed result is in ppo_trader.zip")
