import pymunk
import torch
import random
import cv2
import numpy as np

import matplotlib.pyplot as plt
from ray.rllib.env import MultiAgentEnv
from gymnasium import spaces

from simple_playgrounds.playgrounds.layouts import SingleRoom
from simple_playgrounds.engine import Engine
from simple_playgrounds.agents.agents import BaseAgent
from simple_playgrounds.elements.element import InteractiveElement
from simple_playgrounds.elements.collection.basic import Physical
from simple_playgrounds.agents.parts.actuators import ContinuousActuator
from simple_playgrounds.common.definitions import ElementTypes, CollisionTypes
from simple_playgrounds.agents.parts.controllers import External
from simple_playgrounds.common.texture import ColorTexture, UniqueCenteredStripeTexture
from simple_playgrounds.common.position_utils import CoordinateSampler
from simple_playgrounds.agents.sensors.topdown_sensors import TopdownSensor

from abc import ABC
from typing import Optional, Union

from simple_playgrounds.configs.parser import parse_configuration



class VisibleZoneElement(InteractiveElement, ABC):
    """Base Class for Contact Entities"""

    def __init__(self, **entity_params):

        InteractiveElement.__init__(
            self, visible_shape=True, invisible_shape=True, traversable=False, **entity_params
        )
        
    def _set_shape_collision(self):
        self.pm_invisible_shape.collision_type = CollisionTypes.CONTACT


class MultiAgentRewardZone(VisibleZoneElement, ABC):
    """
    Reward Zones provide a reward to all the agents in the zone.
    """

    def __init__(
        self,
        reward: float,
        limit: Optional[float] = None,
        config_key: Optional[Union[ElementTypes, str]] = None,
        **entity_params
    ):
        """
        MultiAgentRewardZone entities are invisible zones.
        Provide a reward to the agent which is inside the zone.

        Args:
            **kwargs: other params to configure entity. Refer to Entity class

        Keyword Args:
            reward: Reward provided at each timestep when agent is in the zone
            total_reward: Total reward that the entity can provide during an Episode
        """
        default_config = parse_configuration("element_zone", config_key)
        entity_params = {**default_config, **entity_params}

        super().__init__(reward=reward, **entity_params)

        self._limit = limit
        self._total_reward_provided = 0

    @property
    def reward(self):
        # Provide reward in all steps in all interactions. In RewardZone only provided
        # reward during first step and first interaction, so it doesn't work for more than 1 agent
        rew = self._reward

        if self._limit:
            reward_left = self._limit - self._total_reward_provided

            if abs(rew) > abs(reward_left):
                rew = reward_left

        self._total_reward_provided += rew
        return rew

    @reward.setter
    def reward(self, rew: float):
        self._reward = rew

    def reset(self):
        self._total_reward_provided = 0
        super().reset()

    @property
    def terminate_upon_activation(self):
        return False

    def activate(self, *args):
        return None, None
    


class TreasureHunt(MultiAgentEnv):
    def __init__(self, config):
        super(TreasureHunt, self).__init__()

        self.timelimit = config["timelimit"]
        self.random_assign = config["random_assign"]
        self.seed = config["seed"]
        self.truncated = False
        self.playground = SingleRoom(size=(400, 250), wall_type='light')
        self.agent_ids = set()

        self.engine = Engine(
            playground=self.playground, time_limit=(self.timelimit + 1),  
        )

        self.episodes = 0

        lower_wall = Physical(physical_shape="rectangle", size=(380, 10), 
                        texture=ColorTexture(color=[100, 100, 100], size=(380, 10)), 
                        name="lower_wall",
                        mass=100, movable=False, graspable=False)
        physical_block = Physical(physical_shape="rectangle", size=(160, 160),
                        texture=ColorTexture(color=[100, 100, 100], size=(280, 280)),
                        name="physical_block",
                        mass=100, movable=False, graspable=False)
        tunnel_wall1 = Physical(physical_shape="rectangle", size=(30, 160),
                        texture=ColorTexture(color=[100, 100, 100], size=(30, 280)),
                        name="tunnel_wall1",
                        mass=100, movable=False, graspable=False)
        tunnel_wall2 = Physical(physical_shape="rectangle", size=(30, 160),
                        texture=ColorTexture(color=[100, 100, 100], size=(30, 280)),
                        name="tunnel_wall2",
                        mass=100, movable=False, graspable=False)

        self.playground.add_element(lower_wall, ((200, 200),0))
        self.playground.add_element(physical_block, ((145, 125),0))
        self.playground.add_element(tunnel_wall1, ((240, 125),0))
        self.playground.add_element(tunnel_wall2, ((320, 125),0))

        dummy_agent = BaseAgent(controller=External(),
            radius=12,
            interactive=True,)

        lows = []
        highs = []
        actuators = dummy_agent.controller.controlled_actuators
        for actuator in actuators:
            lows.append(actuator.min)
            highs.append(actuator.max)
        self.action_space = spaces.Box(
            low=np.array(lows).astype(np.float32),
            high=np.array(highs).astype(np.float32),
            dtype=np.float32)
        
        self.observation_space = spaces.Box(high=1, low=0, shape=(64, 64, 3), dtype=np.float32)
        
        self.agent_goal_dict = {}
        self.agent_first_reward_dict = {}


    def process_obs(self):
        obs = {}
        for agent in self._active_agents:
            obs[agent.name] = list(agent.observations.values())[0]
        return obs
    
    def step(self, action_dict):
        self.time_steps += 1
        actions = {}
        if action_dict:
            for agent in self._active_agents:
                agent_action = action_dict.get(agent.name)
                actions[agent] = {}
                actuators = agent.controller.controlled_actuators
                act_idx = 0
                for actuator, act in zip(actuators, agent_action):
                    if isinstance(actuator, ContinuousActuator):
                        actions[agent][actuator] = self.clip_actions(act, act_idx)
                        act_idx += 1
                    else:
                        actions[agent][actuator] = round(self.clip_actions(act, act_idx))
                        act_idx += 1
        self.engine.step(actions)
        self.engine.update_observations()
        observations = self.process_obs()
        rewards, dones, truncated, info = self.compute_reward()
        return observations, rewards, dones, truncated, info
    
    def reset(self, seed=None, options=None):
        if self.episodes > 0:
            for agent in self._active_agents:
                self.playground.remove_agent(agent)
        self.engine.reset()

        self.goal_pos = None
        self.spawn_agents()
        self.spawn_goal()
        info = {}
        self._active_agents = self.playground.agents.copy()

        for agent in self._active_agents:
            self.agent_first_reward_dict[agent.name] = True
        
        self.engine.elapsed_time = 0
        self.episodes += 1
        self.time_steps = 0

        self.engine.update_observations()
        observations = self.process_obs()
        return observations, info
    
    def spawn_agents(self):
        speaker_agent_coordinates = CoordinateSampler((200, 230), area_shape="rectangle", size=(360, 20))
        listener_agent_coordinates = CoordinateSampler((200, 20), area_shape="rectangle", size=(360, 20))
        possible_agent_samplers = [speaker_agent_coordinates, listener_agent_coordinates]

        possible_agent_colors = [(255, 255, 255), (170, 170, 170), (0, 0, 255)]
        agent_dict = {}
        
        self.agent_goal_dict = {}
        self.agent_first_reward_dict = {}
        agent_list = []

        for i in range(2):
            agent = BaseAgent(
            controller=External(),
            radius=15,
            interactive=False, 
            name="agent_{0}".format(i),
            texture=UniqueCenteredStripeTexture(size=10,
                color=possible_agent_colors[i], color_stripe=(0,0,0), size_stripe=4),
            temporary=True)
            #Makes agents traversable
            categories = 2**3
            for p in agent.parts:
                p.pm_visible_shape.filter = pymunk.ShapeFilter(categories)
            agent_dict["agent_{0}".format(i)] = agent
            self.agent_goal_dict["agent_{0}".format(i)] = np.zeros(1, dtype=int) #Legacy code, needed
            self.agent_ids.add("agent_{0}".format(i))
            agent_list.append(agent)

        agent_names = ["agent_0", "agent_1"]
        job_list = []
        if self.random_assign:
            speaker_agent = random.choice(agent_names)
            listener_agent = agent_names[0] if speaker_agent == agent_names[1] else agent_names[1]
        else:
            speaker_agent = agent_names[0]
            listener_agent = agent_names[1]

        job_list.append(speaker_agent)
        job_list.append(listener_agent)
        
        
        for agent, idx in zip(agent_list, range(2)):
            ignore_agents = [agent_ig.parts for agent_ig in agent_list if agent_ig != agent]
            ignore_agents = [agent_part for agent_ls in ignore_agents for agent_part in agent_ls]
            agent.add_sensor(TopdownSensor(agent.base_platform, fov=360, resolution=64, max_range=120, normalize=True))
            if agent.name == job_list[0]:
                self.playground.add_agent(agent, possible_agent_samplers[0], allow_overlapping=True, max_attempts=10)
            else:
                self.playground.add_agent(agent, possible_agent_samplers[1], allow_overlapping=True, max_attempts=10)

    def spawn_goal(self):
        for element in self.playground.elements:
            if isinstance(element, MultiAgentRewardZone):
                self.playground._remove_element_from_playground(element)

        goal = MultiAgentRewardZone(reward=1, 
                                    physical_shape="rectangle",
                                    texture=ColorTexture(color=[255, 0, 0], size=(40, 20)),
                                    size=(40, 20), temporary=True)
        
        possible_positions = (((35, 185), 0), ((280, 185), 0), ((365, 185),0 ))

        if self.goal_pos is not None:
            possible_goal_pos_removed = [pos for pos in possible_positions if pos != self.goal_pos]
        else:
            possible_goal_pos_removed = possible_positions

        self.goal_pos = random.choice(possible_goal_pos_removed)
        self.playground.add_element(goal, self.goal_pos)
    
    def compute_reward(self):
        
        rewards = {}
        dones = {}
        truncateds = {}
        info = {}

        achieved_goal = False
        for agent in self._active_agents:
            if agent.reward >= 1:
                achieved_goal = True

        if achieved_goal:
            reward = 5
        else:
            reward = 0

       
        for agent in self._active_agents:
            rewards[agent.name] = reward
            if self.agent_first_reward_dict[agent.name] and bool(reward):
                self.agent_first_reward_dict[agent.name] = False
                info[agent.name] = {"success": 1.0, "goal_line": 0, "true_goal": self.agent_goal_dict[agent.name]}
            else:
                info[agent.name] = {"success": 0.0, "goal_line": 0, "true_goal": self.agent_goal_dict[agent.name]}

            rewards[agent.name] = reward
            done = self.playground.done or not self.engine.game_on

            truncated = self.playground.done or not self.engine.game_on
            dones[agent.name] = done
            truncateds[agent.name] = truncated
            agent.reward = 0
        
        if achieved_goal:
            self.spawn_goal()
        
        dones["__all__"] = all(dones.values())
        truncateds["__all__"] = all(truncateds.values())
        return rewards, dones, truncateds, info
    
    def clip_actions(self, actions, act_idx):
        return np.clip(actions, self.action_space.low[act_idx], self.action_space.high[act_idx])
    
    def render(self):
         image = self.engine.generate_agent_image(self.playground.agents[1], max_size_pg=400)
         return image
       

    def close(self):
        self.engine.terminate()



class TreasureHuntComm(MultiAgentEnv):
    def __init__(self, config):
        super(TreasureHuntComm, self).__init__()

        self.timelimit = config["timelimit"]
        self.single_reward = config["single_reward"]
        self.seed = config["seed"]
        self.message_len = config["message_length"]
        self.vocab_size = config["vocab_size"]
        self.random_assign = config["random_assign"]
        self.truncated = False
        self.playground = SingleRoom(size=(400, 250), wall_type='light')
        self.agent_ids = set()

        self.engine = Engine(
            playground=self.playground, time_limit=(self.timelimit + 1),  
        )

        self.episodes = 0

        lower_wall = Physical(physical_shape="rectangle", size=(380, 10), 
                        texture=ColorTexture(color=[100, 100, 100], size=(380, 10)), 
                        name="lower_wall",
                        mass=100, movable=False, graspable=False)
        physical_block = Physical(physical_shape="rectangle", size=(160, 160),
                        texture=ColorTexture(color=[100, 100, 100], size=(280, 280)),
                        name="physical_block",
                        mass=100, movable=False, graspable=False)
        tunnel_wall1 = Physical(physical_shape="rectangle", size=(30, 160),
                        texture=ColorTexture(color=[100, 100, 100], size=(30, 280)),
                        name="tunnel_wall1",
                        mass=100, movable=False, graspable=False)
        tunnel_wall2 = Physical(physical_shape="rectangle", size=(30, 160),
                        texture=ColorTexture(color=[100, 100, 100], size=(30, 280)),
                        name="tunnel_wall2",
                        mass=100, movable=False, graspable=False)

        self.playground.add_element(lower_wall, ((200, 200),0))
        self.playground.add_element(physical_block, ((145, 125),0))
        self.playground.add_element(tunnel_wall1, ((240, 125),0))
        self.playground.add_element(tunnel_wall2, ((320, 125),0))

        dummy_agent = BaseAgent(controller=External(),
            radius=12,
            interactive=True,)

        lows = []
        highs = []
        actuators = dummy_agent.controller.controlled_actuators
        for actuator in actuators:
            lows.append(actuator.min)
            highs.append(actuator.max)
        actuators_action_space = spaces.Box(
            low=np.array(lows).astype(np.float32),
            high=np.array(highs).astype(np.float32),
            dtype=np.float32)
        
        message_action_space = spaces.MultiDiscrete([self.vocab_size + 1 for m in range(self.message_len)])
        self.action_space = spaces.Dict({
            "actuators_action_space": actuators_action_space,
            "message_action_space": message_action_space
        })
        
        visual_observation_space = spaces.Box(high=1, low=0, shape=(64, 64, 3), dtype=np.float32)
        message_observation_space = spaces.MultiDiscrete([self.vocab_size + 1 for m in range(self.message_len)])
        self.observation_space = spaces.Dict({
            "visual_observation_space": visual_observation_space,
            "message_observation_space": message_observation_space
        })

        self.agent_first_reward_dict = {}


    def process_obs(self, messages):
        obs = {}
        id = 1
        if len(self._active_agents) > 1:
           for agent in self._active_agents:
                #Allow the agent to send a message every self.message_interval time steps. Very crude method to limit
                #the number of messages sent. Want to have something more emergent in the future.
                obs[agent.name] = {
                    "visual_observation_space":list(agent.observations.values())[0],
                    "message_observation_space": messages[self._active_agents[id].name]
                   }
                id -= 1
        else:
            for agent in self._active_agents:
               obs[agent.name] = {
                "visual_observation_space":list(agent.observations.values())[0],
                "message_observation_space": np.zeros(self.message_len, dtype=int)
               }
        return obs
    
    def step(self, action_dict):
        self.time_steps += 1
        actions = {}
        messages = {}
        info = {}
        if action_dict:
            for agent in self._active_agents:
                messages[agent.name] = action_dict.get(agent.name)["message_action_space"]
                info[agent.name] = {"message": messages[agent.name],
                                  "goal": self.agent_goal_dict[agent.name]}
                agent_action = action_dict.get(agent.name)["actuators_action_space"]
                actions[agent] = {}
                actuators = agent.controller.controlled_actuators
                act_idx = 0
                for actuator, act in zip(actuators, agent_action):
                    if isinstance(actuator, ContinuousActuator):
                        actions[agent][actuator] = self.clip_actions(act, act_idx)
                        act_idx += 1
                    else:
                        actions[agent][actuator] = round(self.clip_actions(act, act_idx))
                        act_idx += 1

        self.engine.step(actions)
        self.engine.update_observations()
        observations = self.process_obs(messages)
        rewards, dones, truncated, info = self.compute_reward()
        return observations, rewards, dones, truncated, info
    
    def reset(self, seed=None, options=None):
        if self.episodes > 0:
            for agent in self._active_agents:
                self.playground.remove_agent(agent)
        self.engine.reset()

        self.goal_pos = None
        self.spawn_agents()
        self.spawn_goal()
        info = {}
        self._active_agents = self.playground.agents.copy()

        init_messages = {}
        for agent in self._active_agents:
            self.agent_first_reward_dict[agent.name] = True
            init_messages[agent.name] = np.zeros(self.message_len, dtype=int)
        
        self.engine.elapsed_time = 0
        self.episodes += 1
        self.time_steps = 0

        self.engine.update_observations()
        observations = self.process_obs(init_messages)
        return observations, info
    
    def spawn_agents(self):
        speaker_agent_coordinates = CoordinateSampler((200, 225), area_shape="rectangle", size=(360, 10))
        listener_agent_coordinates = CoordinateSampler((200, 25), area_shape="rectangle", size=(360, 10))
        possible_agent_samplers = [speaker_agent_coordinates, listener_agent_coordinates]

        possible_agent_colors = [(255, 255, 255), (170, 170, 170), (0, 0, 255)]
        agent_dict = {}
        
        self.agent_goal_dict = {}
        self.agent_first_reward_dict = {}
        agent_list = []

        for i in range(2):
            agent = BaseAgent(
            controller=External(),
            radius=15,
            interactive=False, 
            name="agent_{0}".format(i),
            texture=UniqueCenteredStripeTexture(size=10,
                color=possible_agent_colors[i], color_stripe=(0,0,0), size_stripe=4),
            temporary=True)
            #Makes agents traversable
            categories = 2**3
            for p in agent.parts:
                p.pm_visible_shape.filter = pymunk.ShapeFilter(categories)
            agent_dict["agent_{0}".format(i)] = agent
            self.agent_goal_dict["agent_{0}".format(i)] = np.zeros(1, dtype=int) #Legacy code, needed
            self.agent_ids.add("agent_{0}".format(i))
            agent_list.append(agent)

        agent_names = ["agent_0", "agent_1"]
        job_list = []
        if self.random_assign:
            speaker_agent = random.choice(agent_names)
            listener_agent = agent_names[0] if speaker_agent == agent_names[1] else agent_names[1]
        else:
            speaker_agent = agent_names[0]
            listener_agent = agent_names[1]

        job_list.append(speaker_agent)
        job_list.append(listener_agent)
        
        
        for agent, idx in zip(agent_list, range(2)):
            ignore_agents = [agent_ig.parts for agent_ig in agent_list if agent_ig != agent]
            ignore_agents = [agent_part for agent_ls in ignore_agents for agent_part in agent_ls]
            agent.add_sensor(TopdownSensor(agent.base_platform, fov=360, resolution=64, max_range=120, normalize=True))
            if agent.name == job_list[0]:
                self.playground.add_agent(agent, possible_agent_samplers[0], allow_overlapping=True, max_attempts=10)
            else:
                self.playground.add_agent(agent, possible_agent_samplers[1], allow_overlapping=True, max_attempts=10)

    def spawn_goal(self):
        for element in self.playground.elements:
            if isinstance(element, MultiAgentRewardZone):
                self.playground._remove_element_from_playground(element)

        goal = MultiAgentRewardZone(reward=1, 
                                    physical_shape="rectangle",
                                    texture=ColorTexture(color=[255, 0, 0], size=(40, 20)),
                                    size=(40, 20), temporary=True)
        
        possible_positions = (((35, 185), 0), ((280, 185), 0), ((365, 185),0 ))

        if self.goal_pos is not None:
            possible_goal_pos_removed = [pos for pos in possible_positions if pos != self.goal_pos]
        else:
            possible_goal_pos_removed = possible_positions

        self.goal_pos = random.choice(possible_goal_pos_removed)
        self.playground.add_element(goal, self.goal_pos)
    
    def compute_reward(self):
        
        rewards = {}
        dones = {}
        truncateds = {}
        info = {}
        
        achieved_goal = False
        for agent in self._active_agents:
            if agent.reward >= 1:
                achieved_goal = True

        if achieved_goal:
            reward = 5
        else:
            reward = 0

       
        for agent in self._active_agents:
            rewards[agent.name] = reward
            if self.agent_first_reward_dict[agent.name] and bool(reward):
                self.agent_first_reward_dict[agent.name] = False
                info[agent.name] = {"success": 1.0, "goal_line": 0, "true_goal": self.agent_goal_dict[agent.name]}
            else:
                info[agent.name] = {"success": 0.0, "goal_line": 0, "true_goal": self.agent_goal_dict[agent.name]}

            rewards[agent.name] = reward
            done = self.playground.done or not self.engine.game_on

            truncated = self.playground.done or not self.engine.game_on
            dones[agent.name] = done
            truncateds[agent.name] = truncated
            agent.reward = 0
        
        if achieved_goal:
            self.spawn_goal()
        
        dones["__all__"] = all(dones.values())
        truncateds["__all__"] = all(truncateds.values())
        return rewards, dones, truncateds, info
    
    def clip_actions(self, actions, act_idx):
        return np.clip(actions, self.action_space["actuators_action_space"].low[act_idx], 
                       self.action_space["actuators_action_space"].high[act_idx])   
    
    def render(self):
         image = self.engine.generate_agent_image(self.playground.agents[1], max_size_pg=400)
         return image
       
    def close(self):
        self.engine.terminate()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    config = {"num_landmarks": 1,
              "num_agents": 2,
              "timelimit": 4,
              "coop_chance":0.0,
              "message_length": 3,
              "vocab_size": 3,
              "message_penalty": 0.02,
              "seed": 42,
              "playground_width": 300,
              "playground_height": 300,
              "single_goal": True,
              "single_reward": False,
              "random_assign": False,
              "min_prob": 0.025,
              "max_prob": 0.95,}
    env = TreasureHunt(config)
    print(env.action_space.sample())
    for i in range(1000):
        #print(i)
        #actions = {"agent_0": torch.Tensor(env.action_space.sample()),
        #           "agent_1": torch.Tensor(env.action_space.sample()),}
        #print(actions)
        #obs, rewards, dones, _, info = env.step(actions)
        #print(rewards)
        obs = env.reset()
        img = env.render()
        cv2.imshow('agent', img)
        cv2.waitKey(30)

        for e in range(4):
        #    actions = {"agent_0": {"actuators_action_space": torch.Tensor(env.action_space["actuators_action_space"].sample()),
        #                           "message_action_space": torch.Tensor(env.action_space["message_action_space"].sample())},
        #               "agent_1": {"actuators_action_space": torch.Tensor(env.action_space["actuators_action_space"].sample()),
        #                           "message_action_space": torch.Tensor(env.action_space["message_action_space"].sample())}}

            actions = {"agent_0": torch.Tensor(env.action_space.sample()),
                       "agent_1": torch.Tensor(env.action_space.sample())}

            obs, rewards, dones, _, info = env.step(actions)
            img = env.render()
            cv2.imshow('agent', img)
            cv2.waitKey(30)