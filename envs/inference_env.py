import gymnasium as gym
import numpy as np
from typing import Tuple, Dict

from envs.data_loaders import DiskDataLoader
from envs.statistics_recorder import StatisticsRecorder
from envs.gym_space_builder import GymSpaceBuilderOneWay
from envs.scaler import Scaler

import random
import scipy.stats as ss
import collections
from datetime import datetime


class LearningCryptoEnv(gym.Env):
    def __init__(
        self,
        dataset_name: str = 'dataset',
        leverage: float = 2.,
        episode_max_len: int = 168,
        lookback_window_len: int = 168,
        train_start: list = [7200, 10200, 13200, 16200, 19200],
        train_end: list = [9700, 12700, 15700, 18700, 21741 - 1], 
        test_start: list = [9700, 12700, 15700, 18700],
        test_end: list = [10200, 13200, 16200, 19200], 
        order_size: float = 50.,
        initial_capital: float = 1000.,
        open_fee: float = 0.06e-2,
        close_fee: float = 0.06e-2,
        maintenance_margin_percentage: float = 0.012,
        initial_random_allocated: float = 0,
        regime: str = 'training',
        record_stats: bool = False,
        # cold_start_steps: int = 0
    ):
        self.scaler = Scaler(min_quantile = 0.5, max_quantile = 99.5, scale_coef = initial_capital)

        self.dataset_name = dataset_name
        self.leverage = leverage
        self.episode_max_len = episode_max_len
        self.lookback_window_len = lookback_window_len
        self.order_size = order_size
        self.initial_balance = initial_capital
        self.available_balance = initial_capital
        self.wallet_balance = initial_capital
        self.open_fee = open_fee
        self.close_fee = close_fee
        self.maintenance_margin_percentage = maintenance_margin_percentage

        # episode starts with open long/short position with maximum position_value=initial_random_allocated
        self.initial_random_allocated = initial_random_allocated 
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.regime = regime
        self.record_stats = record_stats
        self.price_array, self.tech_array_total = DiskDataLoader(dataset_name=self.dataset_name).load_dataset()
        # for use_attention or use_lstm
        # self.cold_start_steps = cold_start_steps
        # self.observation_dim = self.tech_array_total.shape[1] + 2 # hardcoded, added number of parameters from exchange
        
        # for custom transformer
        self.observation_dim = (self.tech_array_total.shape[1] + 2) * self.lookback_window_len # hardcoded, added 2 parameters from exchange

        self.reward_realized_pnl_short = 0.
        self.reward_realized_pnl_long = 0.

        self.liquidation = False
        self.episode_maxstep_achieved = False

        # episode start index is random
        if self.regime == "training":
            random_interval = np.random.randint(len(self.train_start))
            self.max_step = self.episode_max_len - 1

            # Episode beginning random sampling
            self.time_absolute = np.random.randint(self.train_start[random_interval], self.train_end[random_interval] - self.max_step - 1)

            ## Sample more recent timesteps more often
            # sample_list = np.linspace(-2, 3, self.train_end[random_interval]-self.train_start[random_interval]-self.max_step)
            # cdf = ss.norm.cdf(sample_list, loc=0, scale=1)
            # self.time_absolute_step_array = np.arange(self.train_start[random_interval], self.train_end[random_interval]-self.max_step)

            # sum_cdf = sum(cdf)
            # self.probability_distribution = [float(i)/sum_cdf for i in cdf]

            # self.time_absolute = np.random.choice(self.time_absolute_step_array, 1, p=self.probability_distribution)[0]

        # episode start index is random
        elif self.regime == "evaluation":
            random_interval = np.random.randint(len(self.test_start))
            self.max_step = self.episode_max_len - 1 # self.test_end[random_interval] - self.test_start[random_interval] - 1
            self.time_absolute = np.random.randint(self.test_start[random_interval], self.test_end[random_interval] - self.max_step - 1)

        # episode start index is constant
        elif self.regime == "backtesting":
            random_interval = 0
            self.max_step = self.episode_max_len - 1 # self.test_end[random_interval] - self.test_start[random_interval] - 1
            self.time_absolute = self.test_start[random_interval]

        else:
            raise ValueError(f"Invalid regime: '{self.regime}'. Allowed values are 'training', 'evaluation', or 'backtesting'.")
        
        self.observation_space, self.action_space = GymSpaceBuilderOneWay(observation_dim=self.observation_dim).get_spaces()

        self._reset_env_state()

    def _reset_env_state(self):
        self.statistics_recorder = StatisticsRecorder(record_statistics=self.record_stats)
        self.state_que = collections.deque(maxlen=self.lookback_window_len)
        self.reset_que = collections.deque(maxlen=self.lookback_window_len * 4) # dataframe to fit scaler is 4 times longer than lookback_window_len

        self.time_relative = 0 # steps played in the current episode
        self.wallet_balance = self.initial_balance

        self.liquidation = False
        self.episode_maxstep_achieved = False

        # fixed bid/ask spread
        self.price_bid = self.price_array[self.time_absolute, 0] * (1 - self.open_fee)
        self.price_ask = self.price_array[self.time_absolute, 0] * (1 + self.open_fee)

        # open position at the episode start can be only long or only short
        random_initial_position = random.choice([True, False]) # used if self.initial_random_allocated > 0
        self._reset_env_state_short(random_initial_position)
        self._reset_env_state_long(not random_initial_position) # invert random_initial_position

        self.margin_short, self.margin_long = self._calculate_margin_isolated()

        self.available_balance = max(self.wallet_balance - self.margin_short - self.margin_long, 0)

        # episode start index is random
        if self.regime == "training":
            random_interval = np.random.randint(len(self.train_start))
            self.max_step = self.episode_max_len - 1

            # Episode beginning random sampling
            self.time_absolute = np.random.randint(self.train_start[random_interval], self.train_end[random_interval] - self.max_step - 1)

            ## Sample more recent timesteps more often
            # sample_list = np.linspace(-2, 3, self.train_end[random_interval]-self.train_start[random_interval]-self.max_step)
            # cdf = ss.norm.cdf(sample_list, loc=0, scale=1)
            # self.time_absolute_step_array = np.arange(self.train_start[random_interval], self.train_end[random_interval]-self.max_step)

            # sum_cdf = sum(cdf)
            # self.probability_distribution = [float(i)/sum_cdf for i in cdf]

            # self.time_absolute = np.random.choice(self.time_absolute_step_array, 1, p=self.probability_distribution)[0]

        # episode start index is random
        elif self.regime == "evaluation":
            random_interval = np.random.randint(len(self.test_start))
            self.max_step = self.episode_max_len - 1 # self.test_end[random_interval] - self.test_start[random_interval] - 1
            self.time_absolute = np.random.randint(self.test_start[random_interval], self.test_end[random_interval] - self.max_step - 1)

        # episode start index is condtant
        elif self.regime == "backtesting":
            random_interval = 0
            self.max_step = self.episode_max_len - 1 # self.test_end[random_interval] - self.test_start[random_interval] - 1
            self.time_absolute = self.test_start[random_interval]

        else:
            raise ValueError(f"Invalid regime: '{self.regime}'. Allowed values are 'training', 'evaluation', or 'backtesting'.")

        self.unrealized_pnl_short = (-self.coins_short * (self.average_price_short - self.price_ask)) #- self.fee_to_close_short 
        self.unrealized_pnl_long = (self.coins_long * (self.price_bid - self.average_price_long)) #- self.fee_to_close_long

        # equity at the episode's beginning
        self.equity = self.wallet_balance + self.unrealized_pnl_short + self.unrealized_pnl_long

    # self.coins_short is negative
    def _reset_env_state_short(self, random_open_position):
        # Start episode with already open SHORT position
        if self.regime == "training" and random_open_position:
            # sample average_price from past 24 hours
            self.average_price_short = random.uniform(self.price_array[self.time_absolute - 24, 0], self.price_array[self.time_absolute, 0]) 
            self.position_value_short = random.uniform(0., self.initial_random_allocated)
            self.coins_short = self.position_value_short / self.average_price_short * (-1)
        else:
            self.average_price_short = self.price_array[self.time_absolute, 0]
            self.position_value_short = 0.
            self.coins_short = 0.

    def _reset_env_state_long(self, random_open_position):
        # Start episode with already open LONG position
        if self.regime == "training" and random_open_position:
            # sample average_price from past 24 hours
            self.average_price_long = random.uniform(self.price_array[self.time_absolute - 24, 0], self.price_array[self.time_absolute, 0])
            self.position_value_long = random.uniform(0., self.initial_random_allocated)
            self.coins_long = self.position_value_long / self.average_price_long
        else:
            self.average_price_long = self.price_array[self.time_absolute, 0]
            self.position_value_long = 0.
            self.coins_long = 0.

    def reset(self, seed=7, options={}):
        self._reset_env_state()
        state_array, reset_array = self._get_observation_reset()
        scaled_obs_reset = self.scaler.reset(state_array, reset_array).flatten()

        # return scaled_obs_reset
        return scaled_obs_reset, {}

    def step(self, action: int):
        assert action in [0, 1, 2, 3], action
        ## prevent random actions with not initialized LSTM hidden state, applied if "use_lstm" in PPO config
        # if self.time_relative < self.cold_start_steps:
        #     action = 0

        # price = self.price_array[self.time_absolute, 0]
        self.price_bid = self.price_array[self.time_absolute, 0] * (1 - self.open_fee)
        self.price_ask = self.price_array[self.time_absolute, 0] * (1 + self.open_fee)

        margin_short_start = self.margin_short
        margin_long_start = self.margin_long

        self.reward_realized_pnl_short = 0.
        self.reward_realized_pnl_long = 0.

        # Oneway actions
        if action == 0:  # do nothing
            self.reward_realized_pnl_long = 0.
            self.reward_realized_pnl_short = 0.

        # similar to "BUY" button
        if action == 1: # open/increace long position by self.order_size
            if self.coins_long >= 0:
                if self.available_balance > self.order_size:
                    buy_num_coins = self.order_size / self.price_ask
                    self.average_price_long = (self.position_value_long + buy_num_coins * self.price_ask) / (self.coins_long + buy_num_coins)
                    self.initial_margin_long += buy_num_coins * self.price_ask / self.leverage
                    self.coins_long += buy_num_coins

            if -self.coins_short > 0: # close/decreace short position by self.order_size
                buy_num_coins = min(-self.coins_short, self.order_size / self.price_ask)
                self.initial_margin_short *= min((-self.coins_short - buy_num_coins), 0.) / -self.coins_short
                self.coins_short = min(self.coins_short + buy_num_coins, 0) # cannot be positive
                realized_pnl = buy_num_coins * (self.average_price_short - self.price_ask)  # buy_num_coins is positive
                self.wallet_balance += realized_pnl
                self.reward_realized_pnl_short = realized_pnl

        # similar to "SELL" button
        if action == 2: # close/reduce long position by self.order_size
            if self.coins_long > 0:
                sell_num_coins = min(self.coins_long, self.order_size / self.price_ask)
                self.initial_margin_long *= (max((self.coins_long - sell_num_coins), 0.) / self.coins_long)
                self.coins_long = max(self.coins_long - sell_num_coins, 0) # cannot be negative
                realized_pnl = sell_num_coins * (self.price_bid - self.average_price_long)
                self.wallet_balance += realized_pnl
                self.reward_realized_pnl_long = realized_pnl

            if -self.coins_short >= 0: # open/increase short position by self.order_size
                if (self.available_balance > self.order_size):
                    sell_num_coins = self.order_size / self.price_ask
                    self.average_price_short = (self.position_value_short + sell_num_coins * self.price_bid) / (-self.coins_short + sell_num_coins)
                    self.initial_margin_short += sell_num_coins * self.price_ask / self.leverage
                    self.coins_short -= sell_num_coins

        self.liquidation = -self.unrealized_pnl_long - self.unrealized_pnl_short > self.margin_long + self.margin_short
        self.episode_maxstep_achieved = self.time_relative == self.max_step

        # CLOSE entire position or LIQUIDATION
        if action == 3 or self.liquidation or self.episode_maxstep_achieved:
            # close LONG position
            if self.coins_long > 0:
                sell_num_coins = self.coins_long   
                # becomes zero
                self.initial_margin_long *= max((self.coins_long - sell_num_coins), 0.) / self.coins_long 
                # becomes zero
                self.coins_long = max(self.coins_long - sell_num_coins, 0)
                realized_pnl = sell_num_coins * (self.price_bid - self.average_price_long)
                self.wallet_balance += realized_pnl
                self.reward_realized_pnl_long = realized_pnl

            # close SHORT position
            if -self.coins_short > 0:
                buy_num_coins = -self.coins_short
                # becomes zero
                self.initial_margin_short *= min((self.coins_short + buy_num_coins), 0.) / self.coins_short
                # becomes zero
                self.coins_short += buy_num_coins
                realized_pnl = buy_num_coins * (self.average_price_short - self.price_ask) # buy_num_coins is positive
                self.wallet_balance += realized_pnl
                self.reward_realized_pnl_short = realized_pnl

        self.margin_short, self.margin_long = self._calculate_margin_isolated()
        self.available_balance = max(self.wallet_balance - self.margin_short - self.margin_long, 0)
        self.unrealized_pnl_short = (-self.coins_short * (self.average_price_short - self.price_ask))  # self.coins_short is negatve
        self.unrealized_pnl_long = (self.coins_long * (self.price_bid - self.average_price_long)) # - self.fee_to_close_long
        next_equity = (self.wallet_balance + self.unrealized_pnl_short + self.unrealized_pnl_long)

        done = self.episode_maxstep_achieved or self.liquidation # end of episode or liquidation event

        # reward function
        # normalize rewards to fit [-10:10] range
        reward = (self.reward_realized_pnl_short + self.reward_realized_pnl_long) / self.initial_balance
        # reward = (next_equity - self.equity) / self.initial_balance # reward function for equity changes

        self.equity = next_equity

        margin_short_end = self.margin_short
        margin_long_end = self.margin_long
        
        obs_step = self._get_observation_step(self.time_absolute)
        obs = self.scaler.step(obs_step).flatten()

        self.statistics_recorder.update(
            action=action,
            reward=reward,
            reward_realized_pnl_short=self.reward_realized_pnl_short,
            reward_realized_pnl_long=self.reward_realized_pnl_long,
            unrealized_pnl_short = self.unrealized_pnl_short,
            unrealized_pnl_long = self.unrealized_pnl_long,
            margin_short_start=margin_short_start,
            margin_long_start=margin_long_start,
            margin_short_end=margin_short_end,
            margin_long_end=margin_long_end,
            num_steps=self.time_relative,
            coins_short=self.coins_short,
            coins_long=self.coins_long,
            equity=self.equity,
            wallet_balance=self.wallet_balance,
            average_price_short = self.average_price_short,
            average_price_long = self.average_price_long,
        )

        info = self.statistics_recorder.get()            

        self.time_absolute += 1
        self.time_relative += 1

        return obs, reward, done, False, info

    def _get_observation_reset(self):
        for current_time_absolute in range(self.time_absolute - self.lookback_window_len * 4, self.time_absolute):
            self._get_observation_step(current_time_absolute)

        return np.array(self.state_que), np.array(self.reset_que)

    def _get_observation_step(self, current_time):
        input_array = self.tech_array_total[current_time]

        day_column = input_array[0]
        hour_column = input_array[1]
        available_balance = self.available_balance
        unrealized_pnl = self.unrealized_pnl_long + self.unrealized_pnl_short

        current_observation = np.hstack((day_column, hour_column, available_balance, unrealized_pnl, input_array[2:])).astype(np.float32)
        self.state_que.append(current_observation)
        self.reset_que.append(current_observation)

        return np.array(self.state_que)

    def _calculate_margin_isolated(self):
        self.position_value_short = -self.coins_short * self.average_price_short
        self.position_value_long = self.coins_long * self.average_price_long

        self.initial_margin_short = self.position_value_short / self.leverage
        self.initial_margin_long = self.position_value_long / self.leverage

        self.fee_to_close_short = self.position_value_short * self.close_fee
        self.fee_to_close_long = self.position_value_long * self.close_fee

        self.margin_short = self.initial_margin_short + self.maintenance_margin_percentage * self.position_value_short + self.fee_to_close_short
        self.margin_long = self.initial_margin_long + self.maintenance_margin_percentage * self.position_value_long + self.fee_to_close_long

        return self.margin_short, self.margin_long


if __name__ == "__main__":
    env = LearningCryptoEnv()
    obs = env.reset()