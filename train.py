import ray
from ray import tune
from config import ppo_config # for One-way strategy
# from config_long import ppo_config # for Long only strategy

ray.shutdown()
ray.init()

tune.run(
    "PPO",
    stop={"timesteps_total": int(1e10)},
    config=ppo_config,
    local_dir="./results", # default folder "~ray_results" 
    checkpoint_freq=12,
    checkpoint_at_end=False,
    keep_checkpoints_num=None,
    verbose=2,
    reuse_actors=False,
    # resume=True,
    # restore="./results/PPO/PPO_CryptoEnv_1a171_00000_0_2024-05-02_11-51-01/checkpoint_000012"
)