import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd 

class TradingEnv(gym.Env):
    def __init__(self, df, window_size=30, initial_balance=10_000, fee=0.001):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.fee = fee

        self.action_space = spaces.Discrete(3) # 0 = CELL, 1 = HOLD, 2 = BUY

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size * 3,),
            dtype=np.float32
        )

        self.reset()
    
    # return features for past 30 days
    def _get_obsercation(self):
        frame = self.df.loc[self.current_step - self.window_size:self.current_step - 1,
                            ["log_return", "sma20", "sma50"]]
        obs = frame.values.flatten()
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
            super().reset(seed=seed)

            self.current_step = self.window_size
            self.balance = self.initial_balance
            self.shares = 0
            self.last_price = self.df.loc[self.current_step, "Close"]
            self.equity = self.balance

            obs = self._get_obsercation()
            info = {"balance": self.balance, "shares": self.shares, "equity": self.equity}
            return obs, info

    def step(self, action):
        price = self.df.loc[self.current_step, "Close"]

        prev_equity = self.balance + self.shares * price

        if action == 2 and self.balance > 0:
            # buying with all money
            self.shares = (self.balance * (1 - self.fee)) / price
            self.balance = 0
        
        if action == 0 and self.shares > 0:
            self.balance = self.shares * price * (1 - self.fee)
            self.shares = 0
        
        self.equity = self.balance + self.shares * price

        reward = (self.equity - prev_equity) / prev_equity

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        obs = self._get_obsercation()
        info = {"balance": self.balance, "shares": self.shares, "equity": self.equity}

        return obs, reward, terminated, truncated, info
    
    def render(self):
        print(f"Step: {self.current_step}, Equity: {self.equity:.2f}")

# testing with random operations
if __name__ == "__main__":
    from features import load_data

    df = pd.read_csv("data/MSFT_train.csv")

    env = TradingEnv(df)

    obs, info = env.reset()
    print("Initial obs shape:", obs.shape)
    print("Initial info:", info)

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Equity={info['equity']:.2f}")
        if terminated:
            print("Episode finished!")
            break