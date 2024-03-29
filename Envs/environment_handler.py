import numpy as np
import gymnasium as gym
from Envs.simple_landmarks import MultiAgentLandmarks, MultiGoalEnv, MultiAgentLandmarksComm
from Envs.coop_crafting import CoopCraftingEnv, CoopCraftingEnvComm
from Envs.test_crafting import TestCraftingEnv
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
import torch
from Utils.train_utils import get_init_tensors

COMM_ENVS = ["MultiAgentLandmarksComm", "CoopCraftingEnvComm"]

class EnvironmentHandler():
    """Handler for the RLLib MultiAgentEnvironementWrapper, which allows for easy vectorization
       of multi agent environments."""
    def __init__(self, config):
        super(EnvironmentHandler, self).__init__()
        self.env_config = config["env_config"]
        self.base_env = self.get_env(self.env_config)
        self.vector_env = self.base_env.to_base_env(self._generate_vectorized_env, num_envs=self.env_config["num_envs"])

        self.action_space = self.vector_env.action_space
        if config["env_config"]["env_name"] in COMM_ENVS:
            self.observation_space = self.base_env.observation_space["visual_observation_space"]
            self.movement_shape = self.base_env.action_space["actuators_action_space"].shape
            self.message_shape = self.base_env.action_space["message_action_space"].shape
            self.action_space_shape = (int(np.prod(self.movement_shape) + np.prod(self.message_shape)),)
            
        else:
            self.observation_space = self.vector_env.observation_space
            self.action_space_shape = self.vector_env.action_space.shape
        

    
    def step(self, inputs):
        if self.env_config["env_name"] in COMM_ENVS:
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
        time_till_end_dict = {}
        tasks_success_rate = {}
        
        for a in range(self.env_config["num_agents"]):
            obs = torch.zeros((1, self.env_config["num_envs"]) + self.observation_space.shape)
            rewards = torch.zeros(1, (self.env_config["num_envs"]))
            dones = torch.ones(1, (self.env_config["num_envs"]))
            successes = torch.zeros(1, (self.env_config["num_envs"]))
            goal_lines = torch.zeros(1, (self.env_config["num_envs"]))
            true_goal = torch.zeros(self.env_config["num_landmarks"], (self.env_config["num_envs"]))
            message = torch.zeros((1, self.env_config["num_envs"], self.env_config["message_length"]))
            landmark_contact = torch.zeros((1, self.env_config["num_envs"]))
            agent_tasks_success_rate = {"activate_landmarks":0.0, "double_activate":0.0, "lemon_hunt":0.0, "crafting":0.0, "in_out_machine":0.0, "dropoff":0.0}

            stages_info = []
            for s in range(self.env_config["stages"] * 3):
                stages_info.append(torch.zeros(1, (self.env_config["num_envs"])))


            time_till_end = torch.zeros((1, self.env_config["num_envs"]))

            for env in obs_in.keys():
                
                if "agent_{0}".format(a) in obs_in[env].keys():
                    if self.env_config["env_name"] in COMM_ENVS:
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
                        for s in range(self.env_config["stages"]):
                            stages_info[s][0][env] = info[env]["agent_{0}".format(a)]["success_stage_{0}".format(s+1)]

                        time_till_end[0][env] = info[env]["agent_{0}".format(a)]["time_till_end"]

                    if "coop_success_stage_1" in info[env]["agent_{0}".format(a)].keys():
                        for s in range(self.env_config["stages"]):
                            stages_info[s+self.env_config["stages"]][0][env] = info[env]["agent_{0}".format(a)]["coop_success_stage_{0}".format(s+1)]
                            stages_info[s+self.env_config["stages"]*2][0][env] = info[env]["agent_{0}".format(a)]["single_success_stage_{0}".format(s+1)]

            obs_dict["agent_{0}".format(a)] = obs
            message_dict["agent_{0}".format(a)] = message
            rewards_dict["agent_{0}".format(a)] = rewards
            dones_dict["agent_{0}".format(a)] = dones
            landmark_contact_dict["agent_{0}".format(a)] = landmark_contact
            time_till_end_dict["agent_{0}".format(a)] = time_till_end
            
            infos_dict["agent_{0}".format(a)] = {"success": successes, "goal_line": goal_lines, "true_goal": true_goal}
            
            if "success_stage_1" in info[env]["agent_{0}".format(a)].keys():
                for s in range(self.env_config["stages"]):
                    infos_dict["agent_{0}".format(a)]["success_stage_{0}".format(s+1)] = stages_info[s]

            if "coop_success_stage_1" in info[env]["agent_{0}".format(a)].keys():
                for s in range(self.env_config["stages"]):
                    infos_dict["agent_{0}".format(a)]["coop_success_stage_{0}".format(s+1)] = stages_info[s+self.env_config["stages"]]
                    infos_dict["agent_{0}".format(a)]["single_success_stage_{0}".format(s+1)] = stages_info[s+self.env_config["stages"]*2]
                
                #Track the success rate of each task
                for task in agent_tasks_success_rate.keys():
                    task_success = 0.0
                    task_sampled = 0.0
                    for env in range(self.env_config["num_envs"]):
                        task_success += info[env]["agent_{0}".format(a)]["task_successes"][task]
                        task_sampled +=  info[env]["agent_{0}".format(a)]["tasks_sampled"][task]
                        if task_sampled > 0:
                            agent_tasks_success_rate[task] = task_success / task_sampled
                        else:
                            agent_tasks_success_rate[task] = 0.0
                    tasks_success_rate["agent_{0}".format(a)] = agent_tasks_success_rate
        
        dones_dict["__all__"] = np.array([dones_in[i]["__all__"] for i in range(self.env_config["num_envs"])])
            
        if self.env_config["env_name"] in COMM_ENVS:
            return obs_dict, message_dict, rewards_dict, dones_dict, landmark_contact_dict, time_till_end_dict, infos_dict, tasks_success_rate
        else:
            return obs_dict, rewards_dict, dones_dict, landmark_contact_dict, time_till_end_dict, infos_dict, tasks_success_rate
    

    def reset_all(self, idx: List[int]) -> Tuple[Dict, Dict]:
        """Takes list of idx for environments to reset and returns a tuple of dicts (observtions, infos)"""
        reset_obs, infos_dict = self.vector_env.try_reset(idx)
        self.vector_env.poll()
        
        contact = {"agent_{0}".format(a): torch.zeros((1, self.env_config["num_envs"])) for a in range(self.env_config["num_agents"])}
        time_till_end = {"agent_{0}".format(a): torch.ones((1, self.env_config["num_envs"])) for a in range(self.env_config["num_agents"])}

        if self.env_config["env_name"] in COMM_ENVS:
            obs_dict = {"agent_{0}".format(a): torch.FloatTensor(
                np.array([reset_obs[i]["agent_{0}".format(a)]["visual_observation_space"].copy() for i in range(self.env_config["num_envs"])])) 
                for a in range(self.env_config["num_agents"])}
            message_dict = {"agent_{0}".format(a): torch.FloatTensor(
                np.array([reset_obs[i]["agent_{0}".format(a)]["message_observation_space"] for i in range(self.env_config["num_envs"])]))
                for a in range(self.env_config["num_agents"])}
            return obs_dict, message_dict, contact, time_till_end, infos_dict
        
        else:
            obs_dict = {"agent_{0}".format(a): torch.FloatTensor(
                np.array([reset_obs[i]["agent_{0}".format(a)].copy() for i in range(self.env_config["num_envs"])])).unsqueeze(dim=0) 
                                            for a in range(self.env_config["num_agents"])}
            return obs_dict, contact, time_till_end, infos_dict
    
    def reset(self, idx):
        reset_obs, infos_dict = self.vector_env.try_reset(idx)
        self.vector_env.poll()

        contact = {"agent_{0}".format(a): torch.zeros((1, 1)) for a in range(self.env_config["num_agents"])}
        time_till_end = {"agent_{0}".format(a): torch.ones((1, 1)) for a in range(self.env_config["num_agents"])}

        if self.env_config["env_name"] in COMM_ENVS:
            obs_dict = {"agent_{0}".format(a): torch.FloatTensor(reset_obs[idx]["agent_{0}".format(a)]["visual_observation_space"].copy()).unsqueeze(dim=0)
                                            for a in range(self.env_config["num_agents"])}
            message_dict = {"agent_{0}".format(a): torch.FloatTensor(reset_obs[idx]["agent_{0}".format(a)]["message_observation_space"]).unsqueeze(dim=0)
                                            for a in range(self.env_config["num_agents"])}
            return obs_dict, message_dict, contact, time_till_end, infos_dict
        
        else:
            obs_dict = {"agent_{0}".format(a): torch.FloatTensor(reset_obs[idx]["agent_{0}".format(a)].copy()).unsqueeze(dim=0)
                                            for a in range(self.env_config["num_agents"])}
            return obs_dict, contact, time_till_end, infos_dict



    def get_env(self, config):
        "Takes in the config and returns the environment"
        if config["env_name"] == "MultiAgentLandmarks":
            return MultiAgentLandmarks(config)
        elif config["env_name"] == "MultiGoalEnv":
            return MultiGoalEnv(config)
        elif config["env_name"] == "CoopCraftingEnv":
            return CoopCraftingEnv(config)
        elif config["env_name"] == "TestCraftingEnv":
            return TestCraftingEnv(config)
        elif config["env_name"] == "CoopCraftingEnvComm":
            return CoopCraftingEnvComm(config)
        else:
            return MultiAgentLandmarksComm(config)
        
    
    def _generate_vectorized_env(self, idx):
        return self.get_env(self.env_config)
    
    def _get_num_envs(self):
        return self.env_config["num_envs"]
    

