from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
from env import TradingEnv

if __name__ == "__main__":
    df = pd.read_csv("data/MSFT_train.csv")

    env = DummyVecEnv([lambda: TradingEnv(df, window_size=30)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        tensorboard_log="./logs/"
    )

    model.learn(total_timesteps=600_000)

    model.save("ppo_trader")

    print("Learning completed! Result is in ppo_trader.zip")
