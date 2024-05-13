import numpy as np
from gymnasium import spaces
# from gym import spaces
# import gymnasium as gym
# from gymnasium import spaces

from typing import Tuple


class GymSpaceBuilderHedge:
    def __init__(self, observation_dim: int = 241) -> None:
        self.observation_dim = observation_dim
        
        self.action_space = spaces.Tuple(
            [spaces.Discrete(4), spaces.Discrete(4)]
        )
        
        self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.observation_dim,),
                dtype=np.float64
            )

    def get_spaces(self) -> Tuple[spaces.Space, spaces.Space]:
        return self.observation_space, self.action_space
    
class GymSpaceBuilderOneWay:
    def __init__(self, observation_dim: int = 168 * 241) -> None:

        self.observation_dim = observation_dim
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.observation_dim,),
                dtype=np.float32
            )

    def get_spaces(self) -> Tuple[spaces.Space, spaces.Space]:
        return self.observation_space, self.action_space
    
class GymSpaceBuilderLong:
    def __init__(self, observation_dim: int = 567) -> None:
        self.observation_dim = observation_dim

        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.observation_dim,),
                dtype=np.float32
            )

    def get_spaces(self) -> Tuple[spaces.Space, spaces.Space]:
        return self.observation_space, self.action_space
    
class GymSpaceBuilderShort:
    def __init__(self, observation_dim: int = 567) -> None:
        self.observation_dim = observation_dim

        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.observation_dim,),
                dtype=np.float32
            )

    def get_spaces(self) -> Tuple[spaces.Space, spaces.Space]:
        return self.observation_space, self.action_space