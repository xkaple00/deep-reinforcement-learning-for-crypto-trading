import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

class Scaler:
    def __init__(self, min_quantile: int = 1, max_quantile: int = 99, scale_coef: float = 1e3) -> None:
        self.transformer = None # <class 'sklearn.preprocessing._data.RobustScaler'>
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile
        self.scale_coef = scale_coef

    def reset(self, state_array, reset_array):
        # don't apply scaler to day, hour, unrealized_pnl and available_balance columns
        self.transformer = RobustScaler(quantile_range=(self.min_quantile, self.max_quantile)).fit(reset_array[:, 4:]) 
        scaled_np_array = self.step(state_array)

        return scaled_np_array

    def step(self, state_array):
        day_column = state_array[:, [0]] 
        hour_column = state_array[:, [1]]
        available_balance = state_array[:, [2]] / self.scale_coef
        unrealized_pnl = state_array[:, [3]] / self.scale_coef
        transformed_indicators = np.clip(self.transformer.transform(state_array[:, 4:]), a_min=-10., a_max=10.)
        scaled_np_array = np.hstack((day_column, hour_column, available_balance, unrealized_pnl, transformed_indicators)).astype(np.float32)
        return scaled_np_array


if __name__ == "__main__":
    scaler = Scaler()
    obs = scaler.reset()