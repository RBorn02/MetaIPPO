import pymunk
import torch
import random
import cv2
import numpy as np

import matplotlib.pyplot as plt
from ray.rllib.env import MultiAgentEnv
from gymnasium import spaces

from simple_playgrounds.playgrounds.layouts import SingleRoom, LineRooms
from simple_playgrounds.engine import Engine
from simple_playgrounds.agents.agents import BaseAgent
from simple_playgrounds.elements.collection.activable import ActivableByGem
from simple_playgrounds.agents.parts.actuators import ContinuousActuator
from simple_playgrounds.common.definitions import ElementTypes, CollisionTypes
from simple_playgrounds.agents.parts.controllers import Keyboard, External
from simple_playgrounds.common.texture import ColorTexture, UniqueCenteredStripeTexture
from simple_playgrounds.elements.element import GemElement
from simple_playgrounds.common.position_utils import CoordinateSampler
from simple_playgrounds.agents.sensors.topdown_sensors import TopdownSensor, FullPlaygroundSensor





class Diamond(GemElement):
    def __init__(self, chest, physical_shape, radius, texture, name, **kwargs):

        super().__init__(
            config_key=ElementTypes.KEY,
            elem_activated=chest,
            physical_shape=physical_shape,
            radius=radius,
            texture=texture,
            graspable=True,
            movable=True,
            mass=20,
            name=name,
            **kwargs
        )

        self.chest = chest
        self.graspable = True

class Chest(ActivableByGem):
    def __init__(self, radius, physical_shape, texture, name, **kwargs):

        super().__init__(
            config_key=ElementTypes.CHEST,
            radius=radius,
            physical_shape=physical_shape,
            texture=texture,
            movable=True,
            graspable=True,
            mass=20,
            name=name,
            **kwargs
        )

        self.graspable = True

    def _set_pm_collision_type(self):
        for pm_shape in self._pm_shapes:
            pm_shape.collision_type = CollisionTypes.ACTIVABLE_BY_GEM

    def activate(self, activating):
        
        list_remove = None
        elem_add = None

        if activating.elem_activated is self:

            list_remove = [activating, self]

        return list_remove, elem_add

def plt_image(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()

#Plot image using cv2
def cv2_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class LinRoomEnv(MultiAgentEnv):
    def __init__(self, config):
        super(LinRoomEnv, self).__init__()

        self.config = config
        self.num_goals = config["num_landmarks"]
        self.num_agents = config["num_agents"]
        self.timelimit = config["timelimit"]
        self.coop_chance = config["coop_chance"]
        self.seed = config["seed"]
        self.episodes = 0
        self.time_steps = 0
        self.truncated = False
        self.playground = SingleRoom(size=(400, 150), wall_depth=10)
        self.agent_ids = set()
        self.single_goal = config["single_goal"]
        self.single_reward = config["single_reward"]
        
        self.engine = Engine(
            playground=self.playground, time_limit=(self.timelimit + 1),  
        )

        agent_sampler_chest = CoordinateSampler(
            (360, 75), area_shape="rectangle", size=(80, 130)
        )
        agent_sampler_diamond = CoordinateSampler(
            (60, 75), area_shape="rectangle", size=(80, 130)
        )

        
        possible_agent_colors = [(255, 255, 255), (170, 170, 170), (0, 0, 255)]
        possible_agent_samplers = [agent_sampler_chest, agent_sampler_diamond]
        agent_dict = {}
        
        self.agent_goal_dict = {}
        self.agent_first_reward_dict = {}
        agent_ls = []
        for i in range(self.num_agents):
            agent = BaseAgent(
            controller=External(),
            radius=12,
            interactive=True, 
            name="agent_{0}".format(i),
            texture=UniqueCenteredStripeTexture(size=10,
                color=possible_agent_colors[i], color_stripe=(0,0,0), size_stripe=4))
            #Makes agents traversable
            categories = 2**3
            for p in agent.parts:
                p.pm_visible_shape.filter = pymunk.ShapeFilter(categories)
            agent_dict["agent_{0}".format(i)] = agent
            self.agent_goal_dict["agent_{0}".format(i)] = np.zeros(self.num_goals, dtype=int)
            self.agent_ids.add("agent_{0}".format(i))
            agent_ls.append(agent)
      
        for agent, idx in zip(agent_ls, range(config["num_agents"])):
            ignore_agents = [agent_ig.parts for agent_ig in agent_ls if agent_ig != agent]
            ignore_agents = [agent_part for agent_ls in ignore_agents for agent_part in agent_ls]
            agent.add_sensor(TopdownSensor(agent.base_platform, resolution=64, normalize=True))
            self.playground.add_agent(agent, possible_agent_samplers[idx])

        self.spawn_objects()

        lows = []
        highs = []
        actuators = agent.controller.controlled_actuators
        for actuator in actuators:
            lows.append(actuator.min)
            highs.append(actuator.max)
        self.action_space = spaces.Box(
            low=np.array(lows).astype(np.float32),
            high=np.array(highs).astype(np.float32),
            dtype=np.float32)
        
        self.observation_space = spaces.Box(high=1, low=0, shape=(64, 64, 3), dtype=np.float32)
        
        self._active_agents = self.playground.agents.copy()

    def process_obs(self):
        obs = {}
        for agent in self._active_agents:
            #print(list(agent.observations.values())[0])
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
                        actions[agent][actuator] = int(torch.round(self.clip_actions(act, act_idx)))
                        act_idx += 1
        
        self.engine.step(actions)
        self.engine.update_observations()
        observations = self.process_obs()
        rewards, dones, truncated, info = self.compute_reward()
        return observations, rewards, dones, truncated, info

    def reset(self, seed=None, options=None):
        self.engine.reset()
        info = {}
        self._active_agents = self.playground.agents.copy()
        for agent in self._active_agents:
            self.agent_first_reward_dict[agent.name] = True
        
        self.sample_goal()
        self.engine.elapsed_time = 0
        self.time_steps = 0
        self.episodes += 1

        self.engine.update_observations()
        observations = self.process_obs()
        return observations, info

    def spawn_objects(self):
        chest_coordinates = CoordinateSampler((360, 75), area_shape="rectangle", size=(80, 130))
        diamond_coordinates = CoordinateSampler((40, 75), area_shape="rectangle", size=(80, 130))
        possible_shapes = ["circle", "rectangle", "triangle", "pentagon"]
        possible_colors = [[255,0,0], [0,255,0], [0,0,255],[255,255,0]]
        self.possible_goals = []
        for idx in range(self.config["num_landmarks"]):
            chest = Chest(physical_shape=possible_shapes[idx], radius=10, texture=ColorTexture(color=possible_colors[idx], size=10),
                           name=possible_shapes[idx]+"_chest")
            diamond = Diamond(chest, physical_shape=possible_shapes[idx], radius=10, texture=ColorTexture(color=possible_colors[idx], size=10),
                              name=possible_shapes[idx]+"_diamond")
            self.playground.add_element(chest, chest_coordinates)
            self.playground.add_element(diamond, diamond_coordinates)
            self.possible_goals.append(possible_shapes[idx]+"_chest")


    def sample_goal(self):
        old_goal = None
        
        if old_goal is not None:
            while True:
                self.goal = random.choice(self.possible_goals)
                if old_goal == self.goal:
                    continue
        else:
            self.goal = random.choice(self.possible_goals)

    def compute_reward(self):
        active_element_names = []
        for element in self.playground.elements:
            active_element_names.append(element.name)

        rewards = {}
        dones = {}
        truncateds = {}
        infos = {}
        
        for agent in self._active_agents:
            if self.goal not in active_element_names:
                reward = 1.0
            else:
                reward = 0.0
            rewards[agent.name] = reward

            if self.agent_first_reward_dict[agent.name] and bool(reward):
                self.agent_first_reward_dict[agent.name] = False
                infos[agent.name] = {"success": 1.0, "goal_line": 0.0, "true_goal": self.agent_goal_dict[agent.name]} #Goal doesnt mean anything here, relict of landmarks env
            else:
                infos[agent.name] = {"success": 0.0, "goal_line": 0.0, "true_goal": self.agent_goal_dict[agent.name]}
    
            if self.single_goal:
                if self.single_reward:
                    done = bool(reward) or self.playground.done or not self.engine.game_on
                else:
                    rewards[agent.name] = 0.1 * reward
                    done = self.playground.done or not self.engine.game_on
            else:
                if bool(reward):
                    self.sample_goal()
                done = self.playground.done or not self.engine.game_on
        
            truncated = self.playground.done or not self.engine.game_on
            dones[agent.name] = done
            truncateds[agent.name] = truncated

        dones["__all__"] = all(dones.values())
        truncateds["__all__"] = all(truncateds.values())
        return rewards, dones, truncateds, infos
    
    def clip_actions(self, actions, act_idx):
        return np.clip(actions, self.action_space.low[act_idx], self.action_space.high[act_idx])
    
    def render(self):
         image = self.engine.generate_agent_image(self.playground.agents[1], max_size_pg=400)
         return image
       
    def close(self):
        self.engine.terminate()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    config = {"num_landmarks": 4,
              "num_agents": 2,
              "timelimit": 1000,
              "coop_chance":0.0,
              "message_len": 3,
              "vocab_size": 3,
              "message_penalty": 0.02,
              "seed": 42,
              "single_goal": True,
              "single_reward": False,}
    env = LinRoomEnv(config)
    print(env.action_space.sample())
    obs = env.reset()
    obs_sampled = env.observation_space.sample()
    for i in range(1000):
        print(i)
        actions = {"agent_0": env.action_space.sample(),
                   "agent_1": env.action_space.sample(),}
        print(actions)
        obs, rewards, dones, _, info = env.step(actions)
        print(rewards)
        img = env.render()
        plt.imshow(img)
        plt.show()

