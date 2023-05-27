# Supressing pygame greeting msg
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = ""

import random
import collections
from itertools import product

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from simple_playgrounds.common.texture import ColorTexture, UniqueCenteredStripeTexture
from simple_playgrounds.elements.collection.activable import RewardOnActivation
from simple_playgrounds.agents.parts.actuators import ContinuousActuator
from simple_playgrounds.agents.agents import BaseAgent
from simple_playgrounds.agents.parts.controllers import External
from simple_playgrounds.common.position_utils import CoordinateSampler
from simple_playgrounds.agents.sensors.topdown_sensors import FullPlaygroundSensor
from simple_playgrounds.elements.collection.basic import Wall
from simple_playgrounds.engine import Engine
from simple_playgrounds.playgrounds.layouts import SingleRoom

from abc import ABC
from typing import Optional, Union

from simple_playgrounds.common.definitions import (
    SIMULATION_STEPS,
    CollisionTypes,
    ElementTypes,
)
from simple_playgrounds.configs.parser import parse_configuration
from simple_playgrounds.elements.element import InteractiveElement
from simple_playgrounds.elements.collection.zone import ZoneElement

from ray.rllib.env import MultiAgentEnv

class MultiGoalEnv(gym.Env):
    

    def __init__(self, config):
        super(MultiGoalEnv, self).__init__()

        self.num_goals = config["num_landmarks"]
        self.timelimit = config["timelimit"]
        self.episodes = 0
        self.time_steps = 0
        self.truncated = False

        # Create playground
        # Minimal environment with 1 room and 3 goals
        self.playground = SingleRoom(
            size=(200, 200),
        )
        room = self.playground.grid_rooms[0][0]

        # Only one object will end the episode
        fountain_with_reward = random.choice([n for n in range(self.num_goals)])
        possible_position = [((50, 50), 0), ((100, 50), 0), ((150, 50), 0),
                             ((50, 189), 0), ((100, 189), 0), ((150, 189), 0)]
        for f in range(self.num_goals):
            if f == fountain_with_reward:
                fountain = CustomRewardOnActivation(
                    reward=1, texture = [255, 0, 0], radius = 5,
                    physical_shape = "circle", terminate=True, name=str(f)
                )
            else:
                fountain = CustomRewardOnActivation(
                    reward=0, texture = [255, 0, 0], radius = 5,
                    physical_shape = "circle", terminate=False, name=str(f)
                )
            self.playground.add_element(fountain, possible_position[f])
        self._current_goal = np.zeros(self.num_goals, dtype=np.uint8)
        self._current_goal[fountain_with_reward] = 1
        self.playground._achieved_goal = np.zeros(self.num_goals, dtype=np.uint8)

        # Init the agent
        self.agent = BaseAgent(controller=External(), interactive=True)
        # Add sensor
        self.agent.add_sensor(FullPlaygroundSensor(self.agent, normalize=True))
        # Add agent to playground
        center_area, size_area = room.get_partial_area("down")
        spawn_area_agent = CoordinateSampler(
            center_area, area_shape="rectangle", size=size_area
        )
        self.playground.add_agent(self.agent, spawn_area_agent)

        # Init engine
        self.engine = Engine(
            playground=self.playground, time_limit=self.timelimit
        )

        # Define action and observation space
        # Continuous action space
        actuators = self.agent.controller.controlled_actuators
        lows = []
        highs = []

        for actuator in actuators:

            lows.append(actuator.min)
            highs.append(actuator.max)

        self.action_space = spaces.Box(
            low=np.array(lows).astype(np.float32),
            high=np.array(highs).astype(np.float32),
            dtype=np.float32,
        )
        
        self.observation_space = spaces.Box(high=1, low=0, shape=(64, 64, 3), dtype=np.float32)
        
    def process_obs(self):
        return list(self.agent.observations.values())[0]

    def step(self, action):
        actions_dict = {}
        actuators = self.agent.controller.controlled_actuators
        for actuator, act in zip(actuators, action):
            if isinstance(actuator, ContinuousActuator):
                actions_dict[actuator] = act
            else:
                actions_dict[actuator] = round(act)

        self.engine.step({self.agent: actions_dict})
        self.engine.update_observations()
        done = self.playground.done or not self.engine.game_on
        if self.time_steps >= self.timelimit:
            self.truncated = True
        obs = self.process_obs()

        # Multigoal observations
        achieved_goal = np.zeros(self.num_goals, dtype=np.uint8)
        # deactivate all objects (this should be done in the engine, but its easier to do it
        # like this for now). When an object is pressed it gets activated, so we need to
        # deactivate it afterewards
        for obj in self.playground.elements:
            if isinstance(obj, CustomRewardOnActivation) and obj.activated:
                achieved_goal[int(obj.name)] = 1
                obj.deactivate()
        
        reward = self.compute_reward(achieved_goal, self._current_goal, None)
        info = {"is_success": reward > 0}
        return obs, reward, done, self.truncated, info

    def reset(self, seed=None, options=None):
        self.engine.reset()
        info = {}

        # Sample new goal
        fountain_with_reward = random.choice([g for g in range(self.num_goals)])
        for obj in self.playground.elements:
            if isinstance(obj, CustomRewardOnActivation):
                if obj.name == str(fountain_with_reward):
                    obj.change_state(True)
                    self._current_goal[int(obj.name)] = 1
                else:
                    obj.change_state(False)
                    self._current_goal[int(obj.name)] = 0

        self.engine.elapsed_time = 0
        self.episodes += 1
        self.engine.update_observations()
        self.time_steps = 0
        obs = self.process_obs()
        return obs, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        if np.array_equal(achieved_goal, desired_goal):
            return np.array(5, dtype=np.float32)
        else:
            return np.array(0, dtype=np.float32)

    def render(self, mode=None):
        return (255 * self.engine.generate_agent_image(self.agent)).astype(np.uint8)

    def close(self):
        self.engine.terminate()
        

class MultiAgentLandmarks(MultiAgentEnv):
    def __init__(self, config):
        super(MultiAgentLandmarks, self).__init__()
        
        self.num_goals = config["num_landmarks"]
        self.num_agents = config["num_agents"]
        self.timelimit = config["timelimit"]
        self.coop_chance = config["coop_chance"]
        self.seed = config["seed"]
        self.episodes = 0
        self.time_steps = 0
        self.truncated = False
        self.playground = SingleRoom(size=(200, 200))
        self.goal_space = self._create_goal_space()
        self.agent_ids = set()
        self.single_goal = config["single_goal"]
        self.single_reward = config["single_reward"]
        
        self.engine = Engine(
            playground=self.playground, time_limit=(self.timelimit + 1)
        )
        
        
        possible_positions = [((30, 20),0), ((30, 180),0), ((170, 20),0), ((170, 180), 0),
                              ((30, 100), 0), ((170, 100), 0)]
        possible_textures = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
                             [0, 255, 255]]
        rewards = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10]
        
        for i in range(self.num_goals):
            zone = MultiAgentRewardZone(
            reward=rewards[i],
            physical_shape="rectangle",
            texture=possible_textures[i],
            size=(60, 40))
            #goal_dict["zone_{0}".format(i)] = zone
            self.playground.add_element(zone, possible_positions[i])
        
        agent_sampler = CoordinateSampler(
            (100, 100), area_shape="rectangle", size=(50, 50)
        )
        
        possible_agent_colors = [(255, 255, 255), (170, 170, 170), (0, 0, 255)]
        agent_dict = {}
        
        self.agent_goal_dict = {}
        self.agent_first_reward_dict = {}
        agent_ls = []
        for i in range(self.num_agents):
            agent = BaseAgent(
            #radius=10,
            controller=External(),
            interactive=False, #Agent doesn't need to activate anything
            name="agent_{0}".format(i),
            texture=UniqueCenteredStripeTexture(size=10,
                color=possible_agent_colors[i], color_stripe=possible_agent_colors[i], size_stripe=4))
            agent_dict["agent_{0}".format(i)] = agent
            self.agent_goal_dict["agent_{0}".format(i)] = np.zeros(self.num_goals, dtype=int)
            self.agent_ids.add("agent_{0}".format(i))
            agent_ls.append(agent)
        #Test mode for now!!! Agent dont see each other    
        for agent in agent_ls:
            ignore_walls = [elem for elem in self.playground.elements if isinstance(elem, Wall)]
            ignore_agents = [agent_ig.parts for agent_ig in agent_ls if agent_ig != agent]
            ignore_agents = [agent_part for agent_ls in ignore_agents for agent_part in agent_ls]
            agent.add_sensor(FullPlaygroundSensor(agent, normalize=True))
            self.playground.add_agent(agent, agent_sampler)

            
        actuators = agent.controller.controlled_actuators
        lows = []
        highs = []
        for actuator in actuators:
            lows.append(actuator.min)
            highs.append(actuator.max)
        self.action_space = spaces.Box(
            low=np.array(lows).astype(np.float32),
            high=np.array(highs).astype(np.float32),
            dtype=np.float32)
        
        self.observation_space = spaces.Box(high=1, low=0, shape=(64, 64, 3), dtype=np.float32)
        
        self._active_agents = self.playground.agents.copy()
        
        
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
        self.engine.reset()
        info = {}
        self._active_agents = self.playground.agents.copy()
        if self.coop_chance > np.random.uniform():
            self.episode_coop = True
        else:
            self.episode_coop = False
        for agent in self._active_agents:
            self.agent_first_reward_dict[agent.name] = True
        needs_goal = [agent.name for agent in self.playground.agents]
        self.sample_goals(needs_goal, reset=True)
        self.engine.elapsed_time = 0
        self.episodes += 1
        self.engine.update_observations()
        self.time_steps = 0
        observations = self.process_obs()
        return observations, info
    
    def process_obs(self):
        obs = {}
        for agent in self._active_agents:
            #print(agent.observations.values()[0])
            obs[agent.name] = list(agent.observations.values())[0]
        return obs
    
    def compute_reward(self):
        
        individual_achieved_goals = {}
        for i in range(self.num_agents):
            individual_achieved_goals["agent_{0}".format(i)] = np.zeros(self.num_goals, dtype=int)
        rewards = {}
        dones = {}
        truncateds = {}
        info = {}
        achieved_goal_list = []
        needs_new_goal = []
        for agent in self._active_agents:
            self.check_achieved_goal(agent, individual_achieved_goals)
            achieved_goal_list.append(individual_achieved_goals[agent.name])
            #info[agent.name] = [self.agent_goal_dict[agent.name], individual_achieved_goals[agent.name]]
        
        collective_achieved_goal = np.bitwise_or.reduce(achieved_goal_list, axis=0)
        
        for agent in self._active_agents:
            if (
                np.sum(self.agent_goal_dict[agent.name]) > 1
                and np.all(self.agent_goal_dict[agent.name] == collective_achieved_goal)
            ) or (np.all(self.agent_goal_dict[agent.name] == individual_achieved_goals[agent.name])):
                reward = 1
            else:
                reward = 0
            if (
                np.sum(self.agent_goal_dict[agent.name]) > 1
                and np.all(self.agent_goal_dict[agent.name] == collective_achieved_goal)):
                print("collective goal achieved")
            rewards[agent.name] = reward
            if self.agent_first_reward_dict[agent.name] and bool(reward):
                self.agent_first_reward_dict[agent.name] = False
                info[agent.name] = 1.0
            else:
                info[agent.name] = 0.0

            if self.single_goal:
                if self.single_reward:
                    done = bool(reward) or self.playground.done or not self.engine.game_on
                else:
                    rewards[agent.name] = 0.1 * reward
                    done = self.playground.done or not self.engine.game_on
            else:
                if bool(reward):
                   needs_new_goal.append(agent.name)
                done = self.playground.done or not self.engine.game_on

            truncated = self.playground.done or not self.engine.game_on
            dones[agent.name] = done
            truncateds[agent.name] = truncated
            # logging which goal line the agent achieved (-1 means no goal line)
            # info[agent.name] = {"goal_line": agent.reward - 1}
        
        [
            self._active_agents.remove(agent)
            for agent in self._active_agents
            if dones[agent.name]
        ]
        if len(needs_new_goal) >= 1:
            self.sample_goals(needs_new_goal)
        dones["__all__"] = all(dones.values())
        truncateds["__all__"] = all(truncateds.values())
        return rewards, dones, truncateds, info
    
    def check_achieved_goal(self, agent, individual_achieved_goals):
        if agent.reward:
            if agent.reward < 100:
                agent.reward = 1
            elif agent.reward < 10_000:
                agent.reward = 2
            elif agent.reward < 1_000_000:
                agent.reward = 3
            elif agent.reward < 100_000_000:
                agent.reward = 4
            elif agent.reward < 10_000_000_000:
                agent.reward = 5
            else:
                agent.reward = 6
            individual_achieved_goals[agent.name][agent.reward - 1] = 1
    
    def sample_goals(self, needs_goal, reset=False):
        if self.episode_coop:
            possible_goals = self.goal_space[(self.num_agents-1)*self.num_goals:]
            if reset is False:
                possible_goals.remove(tuple(self.agent_goal_dict["agent_0"]))
                goal = random.choice(possible_goals)
            else:
                goal = random.choice(possible_goals)
            for agent in self.playground.agents:
                self.agent_goal_dict[agent.name] = np.array(goal, dtype=int)
        else:
            last_goal_idx = (len(self.goal_space) - (self.num_agents-1)*self.num_goals)
            #Handle case for just one agent
            if last_goal_idx == len(self.goal_space):
                possible_goals = self.goal_space
            else:
                possible_goals = [goal for goal in self.goal_space if sum(goal) < self.num_agents]
                possible_single_goals = self.goal_space[:self.num_goals]
                
            agents = needs_goal.copy()
            if 1 / self.num_agents > np.random.uniform():
               random.shuffle(agents)
            for agent in agents:
                #print(agent)
                if agent in needs_goal:
                    needs_goal.remove(agent)
                    agent_possible_goals = possible_goals.copy()
                    if reset is False:
                        agent_possible_goals.remove(tuple(self.agent_goal_dict[agent]))
                        goal = random.choice(agent_possible_goals)
                    else:
                        goal = random.choice(agent_possible_goals)
                    if sum(goal) > 1 and len(needs_goal) > 1:
                        second_agent = random.choice(needs_goal)
                        needs_goal.remove(second_agent)
                        self.agent_goal_dict[agent] = np.array(goal, dtype=int)
                        self.agent_goal_dict[second_agent] = np.array(goal, dtype=int)
                        possible_goals.remove(goal)
                    else:
                        agent_possible_single_goals = possible_single_goals.copy()
                        if reset is False:
                            agent_possible_single_goals.remove(tuple(self.agent_goal_dict[agent]))
                            goal = random.choice(agent_possible_goals)
                        else:
                            goal = random.choice(agent_possible_single_goals)
                        self.agent_goal_dict[agent] = np.array(goal, dtype=int)
                        #possible_single_goals.remove(goal)
    
    def clip_actions(self, actions, act_idx):
        return np.clip(actions, self.action_space.low[act_idx], self.action_space.high[act_idx])
    
    def render(self):
         image = self.engine.generate_agent_image(self.playground.agents[1])
         return image
       

    def close(self):
        self.engine.terminate()

    def _create_goal_space(self):
        vectors = set()
        for i in range(1, self.num_agents+1):
            for combination in product(range(self.num_goals), repeat=i):
                vector = [0]*self.num_goals
                for index in combination:
                    vector[index] = 1
                vectors.add(tuple(vector))
        return sorted(list(vectors), key=lambda x: sum(x))
    


class MultiAgentLandmarksComm(MultiAgentEnv):
    def __init__(self, config):
        super(MultiAgentLandmarksComm, self).__init__()
        print(config)
        
        self.num_goals = config["num_landmarks"]
        self.num_agents = config["num_agents"]
        self.timelimit = config["timelimit"]
        self.coop_chance = config["coop_chance"]
        self.message_len = config["message_len"]
        self.vocab_size = config["vocab_size"]
        self.seed = config["seed"]
        self.message_penalty = config["message_penalty"]
        self.episodes = 0
        self.time_steps = 0
        self.truncated = False
        self.playground = SingleRoom(size=(200, 200))
        self.goal_space = self._create_goal_space()
        self.agent_ids = set()
        
        self.engine = Engine(
            playground=self.playground, time_limit=self.timelimit
        )
        
        
        possible_positions = [((30, 20),0), ((30, 180),0), ((170, 20),0), ((170, 180), 0),
                              ((30, 100), 0), ((170, 100), 0)]
        possible_textures = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
                             [0, 255, 255]]
        rewards = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10]
        
        for i in range(self.num_goals):
            zone = MultiAgentRewardZone(
            reward=rewards[i],
            physical_shape="rectangle",
            texture=possible_textures[i],
            size=(40, 20))
            #goal_dict["zone_{0}".format(i)] = zone
            self.playground.add_element(zone, possible_positions[i])
        
        agent_sampler = CoordinateSampler(
            (100, 100), area_shape="rectangle", size=(50, 50)
        )
        
        possible_agent_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        agent_dict = {}
        
        self.agent_goal_dict = {}
        for i in range(self.num_agents):
            agent = BaseAgent(
            controller=External(),
            interactive=False, #Agent doesn't need to activate anything
            name="agent_{0}".format(i),
            texture=UniqueCenteredStripeTexture(size=4,
                color=possible_agent_colors[i], color_stripe=possible_agent_colors[-i], size_stripe=4))
            agent.add_sensor(FullPlaygroundSensor(agent, normalize=True))
            agent_dict["agent_{0}".format(i)] = agent
            self.agent_goal_dict["agent_{0}".format(i)] = np.zeros(self.num_goals, dtype=int)
            self.playground.add_agent(agent, agent_sampler)
            self.agent_ids.add("agent_{0}".format(i))
            
        actuators = agent.controller.controlled_actuators
        lows = []
        highs = []
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
        
        self._active_agents = self.playground.agents.copy()
        
        
    def step(self, action_dict):
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
                for actuator, act in zip(actuators, agent_action):
                    if isinstance(actuator, ContinuousActuator):
                        actions[agent][actuator] = act
                    else:
                        actions[agent][actuator] = round(act)
        self.engine.step(actions)
        self.engine.update_observations()
        observations = self.process_obs(messages)
        rewards, dones, truncated, info = self.compute_reward(messages, info)
        #print(dones)
        #if len(self._active_agents) < 2:
        #    print(rewards, observations)
        self.time_steps += 1
        return observations, rewards, dones, truncated, info

    def reset(self, seed=None, options=None):
        self.engine.reset()
        info = {}
        self._active_agents = self.playground.agents.copy()
        self.agent_penalty_dict = {}
        for agent in self._active_agents:
            info[agent.name] = {}
            self.agent_penalty_dict[agent.name] = 0
        init_messages = {}
        for agent in self._active_agents:
            init_messages[agent.name] = np.zeros(self.message_len, dtype=int)
        self.sample_goals()
        self.engine.elapsed_time = 0
        self.episodes += 1
        self.engine.update_observations()
        self.time_steps = 0
        observations = self.process_obs(init_messages)
        return observations, info
    
    def process_obs(self, messages):
        obs = {}
        id = self.num_agents - 1
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
    
    def compute_reward(self, messages, info):
        individual_achieved_goals = {}
        for i in range(self.num_agents):
            individual_achieved_goals["agent_{0}".format(i)] = np.zeros(self.num_goals, dtype=int)
        rewards = {}
        dones = {}
        truncateds = {}
        achieved_goal_list = []
        for agent in self._active_agents:
            self.check_achieved_goal(agent, individual_achieved_goals)
            achieved_goal_list.append(individual_achieved_goals[agent.name])
            info[agent.name]["achieved_goal"] = individual_achieved_goals[agent.name]
        
        collective_achieved_goal = np.bitwise_or.reduce(achieved_goal_list, axis=0)
        
        for agent in self._active_agents:
            penalty = self.compute_message_penalty(messages[agent.name])
            self.agent_penalty_dict[agent.name] += penalty
            if (
                np.sum(self.agent_goal_dict[agent.name]) > 1
                and np.all(self.agent_goal_dict[agent.name] == collective_achieved_goal)):
                #reward = max(1.0 - self.message_penalty * self.agent_penalty_dict[agent.name], 0)
                reward = 1.0 - self.message_penalty * penalty
                non_penalty_reward = 1
                print("collective goal achieved")
            
            elif (np.all(self.agent_goal_dict[agent.name] == individual_achieved_goals[agent.name])):
                #reward = max(1.0 - self.message_penalty * self.agent_penalty_dict[agent.name], 0)
                reward = 1.0 - self.message_penalty * penalty
                non_penalty_reward = 1
            else:
                reward = 0 - self.message_penalty * penalty
                non_penalty_reward = 0
            info[agent.name]["non_penalized_reward"] = non_penalty_reward
            info[agent.name]["num_words"] = penalty

            rewards[agent.name] = reward
            done = bool(reward) and reward > 0 or self.playground.done or not self.engine.game_on
            truncated = self.playground.done or not self.engine.game_on
            dones[agent.name] = done
            truncateds[agent.name] = truncated
            # logging which goal line the agent achieved (-1 means no goal line)
            #info[agent.name] = {"goal_line": agent.reward - 1}
        
        [
            self._active_agents.remove(agent)
            for agent in self._active_agents
            if dones[agent.name]
        ]
        dones["__all__"] = all(dones.values())
        truncateds["__all__"] = all(truncateds.values())
        return rewards, dones, truncateds, info
    
    def compute_message_penalty(self, message):
        non_zero_count = 0
        for w in message:
            if w > 0:
                non_zero_count += 1
            else:
                continue
        return non_zero_count
    
    def check_achieved_goal(self, agent, individual_achieved_goals):
        if agent.reward:
            if agent.reward < 100:
                agent.reward = 1
            elif agent.reward < 10_000:
                agent.reward = 2
            elif agent.reward < 1_000_000:
                agent.reward = 3
            elif agent.reward < 100_000_000:
                agent.reward = 4
            elif agent.reward < 10_000_000_000:
                agent.reward = 5
            else:
                agent.reward = 6
            individual_achieved_goals[agent.name][agent.reward - 1] = 1
    
    def sample_goals(self):
        if self.coop_chance > np.random.uniform():
            possible_goals = self.goal_space[(self.num_agents-1)*self.num_goals:]
            goal = random.choice(possible_goals)
            for agent in self.playground.agents:
                self.agent_goal_dict[agent.name] = np.array(goal, dtype=int)
        else:
            last_goal_idx = (len(self.goal_space) - (self.num_agents-1)*self.num_goals)
            #Handle case for just one agent
            if last_goal_idx == len(self.goal_space):
                possible_goals = self.goal_space
            else:
                possible_goals = [goal for goal in self.goal_space if sum(goal) < self.num_agents]
                possible_single_goals = self.goal_space[:self.num_goals]
                
            needs_goal = [agent.name for agent in self.playground.agents]
            agents = needs_goal.copy()
            #random.shuffle(agents)
            for agent in agents:
                #print(agent)
                if agent in needs_goal:
                    needs_goal.remove(agent)
                    goal = random.choice(possible_goals)
                    if sum(goal) > 1 and len(needs_goal) > 1:
                        second_agent = random.choice(needs_goal)
                        needs_goal.remove(second_agent)
                        self.agent_goal_dict[agent] = np.array(goal, dtype=int)
                        self.agent_goal_dict[second_agent] = np.array(goal, dtype=int)
                        possible_goals.remove(goal)
                    else:
                        goal = random.choice(possible_single_goals)
                        self.agent_goal_dict[agent] = np.array(goal, dtype=int)
                        #possible_single_goals.remove(goal)
    
    def render(self):
         image = self.engine.generate_agent_image(self.playground.agents[0])
         return image
       

    def close(self):
        self.engine.terminate()
        
    def _create_goal_space(self):
        vectors = set()
        for i in range(1, self.num_agents+1):
            for combination in product(range(self.num_goals), repeat=i):
                vector = [0]*self.num_goals
                for index in combination:
                    vector[index] = 1
                vectors.add(tuple(vector))
        return sorted(list(vectors), key=lambda x: sum(x))
    
    
class VisibleZoneElement(InteractiveElement, ABC):
    """Base Class for Contact Entities"""

    def __init__(self, **entity_params):

        InteractiveElement.__init__(
            self, visible_shape=True, invisible_shape=True, **entity_params
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
       
            

class CustomRewardOnActivation(RewardOnActivation):
    def __init__(self, terminate: bool, **kwargs):
        super().__init__(**kwargs)
        self.change_state(terminate)

    @property
    def terminate_upon_activation(self):
        return self.terminate

    def activate(self, _):
        self.activated = True
        return None, None

    def deactivate(self):
        self.activated = False

    def change_state(self, terminate: bool):
        """Makes the object activable and reward-providing or the contrary"""
        self.reward = int(terminate)
        self.terminate = terminate
        # Set different color when the object is the activable one
        if terminate:
            self._texture_surface.fill(color=(255, 255, 255))
        else:
            self._texture_surface.fill(color=(0, 255, 0))
            
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
    env = MultiAgentLandmarks(config)
    #print(env.action_space.sample())
    obs = env.reset()
    obs_sampled = env.observation_space.sample()
    for i in range(1000):
        print(i)
        actions = {"agent_0": env.action_space.sample(),
                   "agent_1": env.action_space.sample(),}
    #               "agent_2": env.action_space.sample()}
        print(actions)
        obs, rewards, dones, _, info = env.step(actions)
        #print(dones)
        print(rewards)
        #print(info)
        #print(obs)
        #print(env.agent_goal_dict)
        #print(obs["agent_0"] == obs["agent_1"])
        img = env.render()
        plt.imshow(img)
        plt.show()
    
    #config2 = {"num_goals": 3,
    #          "num_agents": 2,
    #          "timelimit": 100,}
    #env2 = MultiGoalEnv(config2)
    #print(env2.action_space.sample())