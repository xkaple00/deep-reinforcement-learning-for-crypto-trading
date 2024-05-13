import gymnasium as gym
import numpy as np

from ray.tune import registry
from ray.rllib.models.catalog import ModelCatalog

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from envs.training_env_long import LearningCryptoEnv
from models.transformer import TransformerModelAdapter


ModelCatalog.register_custom_model(
    model_name='TransformerModelAdapter',
    model_class=TransformerModelAdapter
)

registry.register_env(
    name='CryptoEnv',
    env_creator=lambda env_config: LearningCryptoEnv(**env_config)
)

ppo_config = (
    PPOConfig()
    # .rl_module(_enable_rl_module_api=False)
    .framework('tf')
    .environment(
        env='CryptoEnv',
        env_config={
            "dataset_name": "dataset",  # .npy files should be in ./data/dataset/
            "leverage": 2, # leverage for perpetual futures
            "episode_max_len": 168 * 2, # train episode length, 2 weeks
            "lookback_window_len": 168, 
            "train_start": [2000, 7000, 12000, 17000, 22000],
            "train_end": [6000, 11000, 16000, 21000, 26000], 
            "test_start": [6000, 11000, 16000, 21000, 26000],
            "test_end": [7000, 12000, 17000, 22000, 29377-1], 
    
            "order_size": 50, # dollars
            "initial_capital": 1000, # dollars
            "open_fee": 0.002e-2, # taker_fee
            "close_fee": 0.002e-2, # taker_fee
            "maintenance_margin_percentage": 0.012, # 1.2 percent
            "initial_random_allocated": 0, # open initial long/short position
            # "training": True,
            "regime": "training",

            "record_stats": False, # True for backtesting
            # "cold_start_steps": 0, # do nothing at the beginning of the episode
        },
        observation_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(183 * 168,),
            dtype=np.float32
        ),
        action_space=gym.spaces.Discrete(4),
        # disable_env_checking=True
    )
    .training(
        gamma=1.,
        lr=5e-5,
        grad_clip=30.,
        # grad_clip_by='global_norm',
        train_batch_size=15 * 1 * 168,
        model={
            "vf_share_layers": False,
            "custom_model": "TransformerModelAdapter",
            "custom_model_config": {
                "d_history_flat": 168 * 183, # 211
                "num_obs_in_history": 168,
                "d_obs": 183, #211,
                "d_time": 2,
                "d_account": 2,
                "d_candlesticks_btc": 34, # TA indicators
                "d_candlesticks_ftm": 34, # TA indicators
                "d_santiment_btc_1h": 30,
                "d_santiment_btc_1d": 26,
                "d_santiment_ftm_1h": 27, 
                "d_santiment_ftm_1d": 28, 
                "d_obs_enc": 256,
                "num_attn_blocks": 3,
                "num_heads": 4,
                "dropout_rate": 0.1
            }
        }

    )
    .evaluation(
        evaluation_interval=1,
        evaluation_duration=8,
        evaluation_duration_unit='episodes',
        evaluation_parallel_to_training=False,

        evaluation_config={
            "explore": False,
            "env_config": {
                # "training": False, 
                "regime": "evaluation",
                "record_stats": False, # True for backtesting
                "episode_max_len": 168 * 2, # validation episode length
                "lookback_window_len": 168, 
            }
        },
        evaluation_num_workers=4
    )
    .rollouts(
        num_rollout_workers=15,
        num_envs_per_worker=1,
        rollout_fragment_length=168,
        batch_mode='complete_episodes',
        preprocessor_pref=None
    )
    .resources(
        num_gpus=1
    )
    .debugging(
        log_level='WARN'
    )
)