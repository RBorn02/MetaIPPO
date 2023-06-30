import numpy as np
import gymnasium as gym
from Envs.simple_landmarks import MultiAgentLandmarks, MultiGoalEnv, MultiAgentLandmarksComm
from Envs.linear_rooms import LinRoomEnv, LinRoomEnvComm, LinLandmarksEnv, LinLandmarksEnvComm
from Envs.crafting_env import CraftingEnv
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
import torch
from Utils.train_utils import get_init_tensors

class EnvironmentHandler():
    """Handler for the RLLib MultiAgentEnvironementWrapper, which allows for easy vectorization
       of multi agent environments."""
    def __init__(self, config):
        super(EnvironmentHandler, self).__init__()
        self.env_config = config["env_config"]
        self.base_env = self.get_env(self.env_config)
        self.vector_env = self.base_env.to_base_env(self._generate_vectorized_env, num_envs=self.env_config["num_envs"])

        self.action_space = self.vector_env.action_space
        if config["env_config"]["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm"]:
            self.observation_space = self.base_env.observation_space["visual_observation_space"]
            self.movement_shape = self.base_env.action_space["actuators_action_space"].shape
            self.message_shape = self.base_env.action_space["message_action_space"].shape
            self.action_space_shape = (int(np.prod(self.movement_shape) + np.prod(self.message_shape)),)
            
        else:
            self.observation_space = self.vector_env.observation_space
            self.action_space_shape = self.vector_env.action_space.shape
        

    
    def step(self, inputs):
        if self.env_config["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm"]:
            movement_actions = inputs["actions"]
            message_actions = inputs["messages"]
            action_message_dict = {e: {"agent_{0}".format(a): {"actuators_action_space": movement_actions[e][a], "message_action_space": message_actions[e][a]} 
                                       for a in range(self.env_config["num_agents"])} for e in range(self.env_config["num_envs"])}
            self.vector_env.send_actions(action_message_dict)
        else:
            actions = inputs
            actions = {e: {"agent_{0}".format(a): actions[e][a] for a in range(self.env_config["num_agents"])} for e in range(self.env_config["num_envs"])}
            self.vector_env.send_actions(actions)
        obs_in, rewards_in, dones_in, truncateds, info = self.vector_env.poll()[:-1]
        #print(obs_in)
        obs_dict = {}
        rewards_dict = {}
        dones_dict = {}
        infos_dict = {}
        message_dict = {}
        landmark_contact_dict = {}
        for a in range(self.env_config["num_agents"]):
            obs = torch.zeros((1, self.env_config["num_envs"]) + self.observation_space.shape)
            rewards = torch.zeros(1, (self.env_config["num_envs"]))
            dones = torch.ones(1, (self.env_config["num_envs"]))
            successes = torch.zeros(1, (self.env_config["num_envs"]))
            goal_lines = torch.zeros(1, (self.env_config["num_envs"]))
            true_goal = torch.zeros(self.env_config["num_landmarks"], (self.env_config["num_envs"]))
            message = torch.zeros((1, self.env_config["num_envs"], self.env_config["message_length"]))
            landmark_contact = torch.zeros((1, self.env_config["num_envs"]))
            stage_one_successes = torch.zeros((1, self.env_config["num_envs"]))
            stage_two_sucecsses = torch.zeros((1, self.env_config["num_envs"]))
            stage_three_successes = torch.zeros((1, self.env_config["num_envs"]))

            for env in obs_in.keys():
                #print(env)
                if "agent_{0}".format(a) in obs_in[env].keys():
                    if self.env_config["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm"]:
                        message[0][env] = torch.Tensor(obs_in[env]["agent_{0}".format(a)]["message_observation_space"])
                        obs[0][env] = torch.Tensor(obs_in[env]["agent_{0}".format(a)]["visual_observation_space"].copy())
                    else:
                        obs[0][env] = torch.Tensor(obs_in[env]["agent_{0}".format(a)].copy())
                    rewards[0][env] = rewards_in[env]["agent_{0}".format(a)]
                    dones[0][env] = 1.0 if dones_in[env]["agent_{0}".format(a)] else 0.0 #Return 1.0 for all steps that agent is done, not only one time
                    successes[0][env] = info[env]["agent_{0}".format(a)]["success"]
                    goal_lines[0][env] = info[env]["agent_{0}".format(a)]["goal_line"]
                    true_goal[:,env] = torch.Tensor(info[env]["agent_{0}".format(a)]["true_goal"]).unsqueeze(dim=0)
                    if info[env]["agent_{0}".format(a)]["goal_line"] == -1:
                        landmark_contact[0][env] = 0.0
                    else:
                        landmark_contact[0][env] = 1.0
                    if "success_stage_1" in info[env]["agent_{0}".format(a)].keys():
                        stage_one_successes[0][env] = info[env]["agent_{0}".format(a)]["success_stage_1"]
                        stage_two_sucecsses[0][env] = info[env]["agent_{0}".format(a)]["success_stage_2"]
                        stage_three_successes[0][env] = info[env]["agent_{0}".format(a)]["success_stage_3"]

            obs_dict["agent_{0}".format(a)] = obs
            message_dict["agent_{0}".format(a)] = message
            rewards_dict["agent_{0}".format(a)] = rewards
            dones_dict["agent_{0}".format(a)] = dones
            landmark_contact_dict["agent_{0}".format(a)] = landmark_contact
            infos_dict["agent_{0}".format(a)] = {"success": successes, "goal_line": goal_lines, "true_goal": true_goal}
            
            if "success_stage_1" in info[env]["agent_{0}".format(a)].keys():
                infos_dict["agent_{0}".format(a)]["success_stage_1"] = stage_one_successes
                infos_dict["agent_{0}".format(a)]["success_stage_2"] = stage_two_sucecsses
                infos_dict["agent_{0}".format(a)]["success_stage_3"] = stage_three_successes

        dones_dict["__all__"] = np.array([dones_in[i]["__all__"] for i in range(self.env_config["num_envs"])]) 
            
        if self.env_config["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm"]:
            return obs_dict, message_dict, rewards_dict, dones_dict, landmark_contact_dict, infos_dict
        else:
            return obs_dict, rewards_dict, dones_dict, landmark_contact_dict, infos_dict
    

    def reset_all(self, idx: List[int]) -> Tuple[Dict, Dict]:
        """Takes list of idx for environments to reset and returns a tuple of dicts (observtions, infos)"""
        reset_obs, infos_dict = self.vector_env.try_reset(idx)
        self.vector_env.poll()
        
        contact = {"agent_{0}".format(a): torch.zeros((1, self.env_config["num_envs"])) for a in range(self.env_config["num_agents"])}

        if self.env_config["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm"]:
            obs_dict = {"agent_{0}".format(a): torch.FloatTensor(
                np.array([reset_obs[i]["agent_{0}".format(a)]["visual_observation_space"].copy() for i in range(self.env_config["num_envs"])])) 
                for a in range(self.env_config["num_agents"])}
            message_dict = {"agent_{0}".format(a): torch.FloatTensor(
                np.array([reset_obs[i]["agent_{0}".format(a)]["message_observation_space"] for i in range(self.env_config["num_envs"])]))
                for a in range(self.env_config["num_agents"])}
            return obs_dict, message_dict, contact, infos_dict
        
        else:
            obs_dict = {"agent_{0}".format(a): torch.FloatTensor(
                np.array([reset_obs[i]["agent_{0}".format(a)].copy() for i in range(self.env_config["num_envs"])])).unsqueeze(dim=0) 
                                            for a in range(self.env_config["num_agents"])}
            return obs_dict, contact, infos_dict
    
    def reset(self, idx):
        reset_obs, infos_dict = self.vector_env.try_reset(idx)
        self.vector_env.poll()

        contact = {"agent_{0}".format(a): torch.zeros((1, 1)) for a in range(self.env_config["num_agents"])}

        if self.env_config["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm"]:
            obs_dict = {"agent_{0}".format(a): torch.FloatTensor(reset_obs[idx]["agent_{0}".format(a)]["visual_observation_space"].copy()).unsqueeze(dim=0)
                                            for a in range(self.env_config["num_agents"])}
            message_dict = {"agent_{0}".format(a): torch.FloatTensor(reset_obs[idx]["agent_{0}".format(a)]["message_observation_space"]).unsqueeze(dim=0)
                                            for a in range(self.env_config["num_agents"])}
            return obs_dict, message_dict, contact, infos_dict
        
        else:
            obs_dict = {"agent_{0}".format(a): torch.FloatTensor(reset_obs[idx]["agent_{0}".format(a)].copy()).unsqueeze(dim=0)
                                            for a in range(self.env_config["num_agents"])}
            return obs_dict, contact, infos_dict



    def get_env(self, config):
        "Takes in the config and returns the environment"
        if config["env_name"] == "MultiAgentLandmarks":
            return MultiAgentLandmarks(config)
        elif config["env_name"] == "MultiGoalEnv":
            return MultiGoalEnv(config)
        elif config["env_name"] == "LinRoomEnv":
            return LinRoomEnv(config)
        elif config["env_name"] == "LinLandmarksEnv":
            return LinLandmarksEnv(config)
        elif config["env_name"] == "CraftingEnv":
            return CraftingEnv(config)
        elif config["env_name"] == "LinRoomEnvComm":
            return LinRoomEnvComm(config)
        elif config["env_name"] == "LinLandmarksEnvComm":
            return LinLandmarksEnvComm(config)
        else:
            return MultiAgentLandmarksComm(config)
        
    
    def _generate_vectorized_env(self, idx):
        return self.get_env(self.env_config)
    
    def _get_num_envs(self):
        return self.env_config["num_envs"]
    

class EnvironmentHandlerPop():
    """Handler for the RLLib MultiAgentEnvironementWrapper, which allows for easy vectorization
       of multi agent environments."""
    def __init__(self, config):
        super(EnvironmentHandlerPop, self).__init__()
        self.env_config = config["env_config"]
        self.base_env = self.get_env(self.env_config)
        self.vector_env = self.base_env.to_base_env(self._generate_vectorized_env, num_envs=self.env_config["num_envs"])

        self.action_space = self.vector_env.action_space
        if config["env_config"]["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm"]:
            self.observation_space = self.base_env.observation_space["visual_observation_space"]
            self.movement_shape = self.base_env.action_space["actuators_action_space"].shape
            self.message_shape = self.base_env.action_space["message_action_space"].shape
            self.action_space_shape = (int(np.prod(self.movement_shape) + np.prod(self.message_shape)),)
            
        else:
            self.observation_space = self.vector_env.observation_space
            self.action_space_shape = self.vector_env.action_space.shape
        

    
    def step(self, inputs, agent_ids):
        if self.env_config["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm"]:
            movement_actions = inputs["actions"]
            message_actions = inputs["messages"]
            action_message_dict = {e: {"agent_{0}".format(a): {"actuators_action_space": movement_actions[e][a], "message_action_space": message_actions[e][a]} 
                                       for a in range(len(agent_ids))} for e in range(self.env_config["num_envs"])}
            self.vector_env.send_actions(action_message_dict)
        else:
            actions = inputs
            actions = {e: {"agent_{0}".format(a): actions[e][a] for a in range(len(agent_ids))} for e in range(self.env_config["num_envs"])}
    
            self.vector_env.send_actions(actions)
        obs_in, rewards_in, dones_in, truncateds, info = self.vector_env.poll()[:-1]
        #print(obs_in)
        obs_dict = {}
        rewards_dict = {}
        dones_dict = {}
        infos_dict = {}
        message_dict = {}
        landmark_contact_dict = {}
        for a, i in zip(agent_ids, range(len(agent_ids))):
            obs = torch.zeros((1, self.env_config["num_envs"]) + self.observation_space.shape)
            rewards = torch.zeros(1, (self.env_config["num_envs"]))
            dones = torch.ones(1, (self.env_config["num_envs"]))
            successes = torch.zeros(1, (self.env_config["num_envs"]))
            goal_lines = torch.zeros(1, (self.env_config["num_envs"]))
            true_goal = torch.zeros(self.env_config["num_landmarks"], (self.env_config["num_envs"]))
            message = torch.zeros((1, self.env_config["num_envs"], self.env_config["message_length"]))
            landmark_contact = torch.zeros((1, self.env_config["num_envs"]))
            for env in obs_in.keys():
                #print(env)
                if "agent_{0}".format(i) in obs_in[env].keys():
                    if self.env_config["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm"]:
                        message[0][env] = torch.Tensor(obs_in[env]["agent_{0}".format(i)]["message_observation_space"])
                        obs[0][env] = torch.Tensor(obs_in[env]["agent_{0}".format(i)]["visual_observation_space"].copy())
                    else:
                        obs[0][env] = torch.Tensor(obs_in[env]["agent_{0}".format(i)].copy())
                    rewards[0][env] = rewards_in[env]["agent_{0}".format(i)]
                    dones[0][env] = 1.0 if dones_in[env]["agent_{0}".format(i)] else 0.0 #Return 1.0 for all steps that agent is done, not only one time
                    successes[0][env] = info[env]["agent_{0}".format(i)]["success"]
                    goal_lines[0][env] = info[env]["agent_{0}".format(i)]["goal_line"]
                    true_goal[:,env] = torch.Tensor(info[env]["agent_{0}".format(i)]["true_goal"]).unsqueeze(dim=0)
                    if info[env]["agent_{0}".format(i)]["goal_line"] == -1:
                        landmark_contact[0][env] = 0.0
                    else:
                        landmark_contact[0][env] = 1.0

            obs_dict["agent_{0}".format(a)] = obs
            message_dict["agent_{0}".format(a)] = message
            rewards_dict["agent_{0}".format(a)] = rewards
            dones_dict["agent_{0}".format(a)] = dones
            landmark_contact_dict["agent_{0}".format(a)] = landmark_contact
            infos_dict["agent_{0}".format(a)] = {"success": successes, "goal_line": goal_lines, "true_goal": true_goal}
        dones_dict["__all__"] = np.array([dones_in[i]["__all__"] for i in range(self.env_config["num_envs"])]) 
            
        if self.env_config["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm"]:
            return obs_dict, message_dict, rewards_dict, dones_dict, landmark_contact_dict, infos_dict
        else:
            return obs_dict, rewards_dict, dones_dict, landmark_contact_dict, infos_dict
    

    def reset_all(self, idx: List[int], agent_ids) -> Tuple[Dict, Dict]:
        """Takes list of idx for environments to reset and returns a tuple of dicts (observtions, infos)"""
        reset_obs, infos_dict = self.vector_env.try_reset(idx)
        self.vector_env.poll()
        
        contact = {"agent_{0}".format(a): torch.zeros((1, self.env_config["num_envs"])) for a in agent_ids}

        if self.env_config["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm"]:
            obs_dict = {"agent_{0}".format(a): torch.FloatTensor(
                np.array([reset_obs[i]["agent_{0}".format(a_in)]["visual_observation_space"].copy() for i in range(self.env_config["num_envs"])])) 
                for a, a_in in zip(agent_ids, range(len(agent_ids)))}
            message_dict = {"agent_{0}".format(a): torch.FloatTensor(
                np.array([reset_obs[i]["agent_{0}".format(a_in)]["message_observation_space"] for i in range(self.env_config["num_envs"])]))
                for a, a_in in zip(agent_ids, range(len(agent_ids)))}
            return obs_dict, message_dict, contact, infos_dict
        
        else:
            obs_dict = {"agent_{0}".format(a): torch.FloatTensor(
                np.array([reset_obs[i]["agent_{0}".format(a_in)].copy() for i in range(self.env_config["num_envs"])])).unsqueeze(dim=0) 
                                            for a, a_in in zip(agent_ids, range(len(agent_ids)))}
            return obs_dict, contact, infos_dict
    
    def reset(self, idx, agent_ids):
        reset_obs, infos_dict = self.vector_env.try_reset(idx)
        self.vector_env.poll()

        contact = {"agent_{0}".format(a): torch.zeros((1, 1)) for a in range(self.env_config["num_agents"])}

        if self.env_config["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm"]:
            obs_dict = {"agent_{0}".format(a): torch.FloatTensor(reset_obs[idx]["agent_{0}".format(a_in)]["visual_observation_space"].copy()).unsqueeze(dim=0)
                                            for a, a_in in zip(agent_ids, range(len(agent_ids)))}
            message_dict = {"agent_{0}".format(a): torch.FloatTensor(reset_obs[idx]["agent_{0}".format(a_in)]["message_observation_space"]).unsqueeze(dim=0)
                                            for a, a_in in zip(agent_ids, range(len(agent_ids)))}
            return obs_dict, message_dict, contact, infos_dict
        
        else:
            obs_dict = {"agent_{0}".format(a): torch.FloatTensor(reset_obs[idx]["agent_{0}".format(a_in)].copy()).unsqueeze(dim=0)
                                            for a, a_in in zip(agent_ids, range(len(agent_ids)))}
            return obs_dict, contact, infos_dict



    def get_env(self, config):
        "Takes in the config and returns the environment"
        if config["env_name"] == "MultiAgentLandmarks":
            return MultiAgentLandmarks(config)
        elif config["env_name"] == "MultiGoalEnv":
            return MultiGoalEnv(config)
        elif config["env_name"] == "LinRoomEnv":
            return LinRoomEnv(config)
        elif config["env_name"] == "LinRoomEnvComm":
            return LinRoomEnvComm(config)
        elif config["env_name"] == "LinLandmarksEnv":
            return LinLandmarksEnv(config)
        elif config["env_name"] == "LinLandmarksEnvComm":
            return LinLandmarksEnvComm(config)
        else:
            return MultiAgentLandmarksComm(config)
        
    
    def _generate_vectorized_env(self, idx):
        return self.get_env(self.env_config)
    
    def _get_num_envs(self):
        return self.env_config["num_envs"]
    

