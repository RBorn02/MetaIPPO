import numpy as np
import gymnasium as gym
from Envs.simple_landmarks import MultiAgentLandmarks, MultiGoalEnv, MultiAgentLandmarksComm
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
import torch

class EnvironmentHandler():
    """Handler for the RLLib MultiAgentEnvironementWrapper, which allows for easy vectorization
       of multi agent environments."""
    def __init__(self, config):
        super(EnvironmentHandler, self).__init__()
        self.env_config = config["env_config"]
        self.base_env = self.get_env(self.env_config)
        self.vector_env = self.base_env.to_base_env(self._generate_vectorized_env, num_envs=self.env_config["num_envs"])

        self.action_space = self.vector_env.action_space
        self.observation_space = self.vector_env.observation_space

    
    def step(self, actions: torch.Tensor) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
        """Takes in a tensor of actions of size (num_envs, num_agents, action_space) returns
           a tuple of dicts (observations, rewards, dones, truncateds, infos)"""
        actions = {e: {"agent_{0}".format(a): actions[e][a] for a in range(self.env_config["num_agents"])} for e in range(self.env_config["num_envs"])}
        self.vector_env.send_actions(actions)
        obs, rewards, dones, truncateds, infos = self.vector_env.poll()[:-1]

        obs_dict = {"agent_{0}".format(a): torch.FloatTensor(
            np.array([obs[i]["agent_{0}".format(a)] for i in range(self.env_config["num_envs"])]))
                                            for a in range(self.env_config["num_agents"])}
        
        rewards_dict = {"agent_{0}".format(a): torch.FloatTensor(
            np.array([rewards[i]["agent_{0}".format(a)] for i in range(self.env_config["num_envs"])]))
                                            for a in range(self.env_config["num_agents"])}
        
        dones_dict = {"agent_{0}".format(a): np.array([dones[i]["agent_{0}".format(a)] for i in range(self.env_config["num_envs"])])
                                            for a in range(self.env_config["num_agents"])}
        dones_dict["__all__"] = np.array([dones[i]["__all__"] for i in range(self.env_config["num_envs"])])
        
        #truncateds_dict = {"agent_{0}".format(a): [truncateds[i]["Agent_{0}".format(a)] for i in range(self.env_config["num_envs"])]
        #                                    for a in range(self.env_config["num_agents"])}
        
        infos_dict = {"agent_{0}".format(a): [infos[i]["agent_{0}".format(a)] for i in range(self.env_config["num_envs"])]
                                            for a in range(self.env_config["num_agents"])}
        
        return obs_dict, rewards_dict, dones_dict, infos_dict
    

    def reset(self, idx: List[int]) -> Tuple[Dict, Dict]:
        """Takes list of idx for environments to reset and returns a tuple of dicts (observations, infos)"""
        reset_obs, infos_dict = self.vector_env.try_reset(idx)
        self.vector_env.poll()

        obs_dict = {"agent_{0}".format(a): torch.FloatTensor(
            np.array([reset_obs[i]["agent_{0}".format(a)] for i in range(self.env_config["num_envs"])])) 
                                            for a in range(self.env_config["num_agents"])}
        return obs_dict, infos_dict


    def get_env(self, config):
        "Takes in the config and returns the environment"
        if config["env_name"] == "MultiAgentLandmarks":
            return MultiAgentLandmarks(config)
        elif config["env_name"] == "MultiGoalEnv":
            return MultiGoalEnv(config)
        elif config["env_name"] == "MultiAgentLandmarksComm":
            return MultiAgentLandmarksComm(config)
        
    
    def _generate_vectorized_env(self, idx):
        return self.get_env(self.env_config)
    
    def _get_num_envs(self):
        return self.env_config["num_envs"]