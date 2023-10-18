import pymunk
import torch
import random
import cv2
import numpy as np

import matplotlib.pyplot as plt
from ray.rllib.env import MultiAgentEnv
from gymnasium import spaces

from simple_playgrounds.playgrounds.layouts import SingleRoom, LineRooms, GridRooms
from simple_playgrounds.engine import Engine
from simple_playgrounds.agents.agents import BaseAgent
from simple_playgrounds.elements.collection.activable import ActivableByGem, RewardOnActivation, ActivableElement
from simple_playgrounds.elements.collection.contact import ContactElement
from simple_playgrounds.elements.collection.basic import Physical, Wall
from simple_playgrounds.agents.parts.actuators import ContinuousActuator
from simple_playgrounds.common.definitions import ElementTypes, CollisionTypes
from simple_playgrounds.agents.parts.controllers import Keyboard, External
from simple_playgrounds.common.texture import ColorTexture, UniqueCenteredStripeTexture, MultipleCenteredStripesTexture
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
            traverseable=True,
            mass=1,
            name=name,
            **kwargs
        )

        self.chest = chest
        self.graspable = True

class Chest(ActivableByGem):
    def __init__(self, radius, physical_shape, texture, name, condition_obj=True, movable=True,
                  graspable=True, out_reward=None, dropoff=False, **kwargs):

        super().__init__(
            config_key=ElementTypes.CHEST,
            radius=radius,
            physical_shape=physical_shape,
            texture=texture,
            movable=movable,
            graspable=graspable,
            traverseable=True,
            mass=1,
            name=name,
            **kwargs
        )

        self.dropoff = dropoff
        self.condition_obj = condition_obj
        if out_reward != "no_object":
            self.out_reward = out_reward
        else:
            self.out_reward = None
        
        self.condition_satisfied = False

    def _set_pm_collision_type(self):
        for pm_shape in self._pm_shapes:
            pm_shape.collision_type = CollisionTypes.ACTIVABLE_BY_GEM

    def activate(self, activating):
        
        list_remove = None
        elem_add = None

        if activating.elem_activated is self:
            self.condition_satisfied = True
            if self.dropoff:
                list_remove = [activating]
            else:
                list_remove = [activating, self]

            if self.condition_obj:
                if self.out_reward is not None:
                    elem_add = [(self.out_reward, self.coordinates)]


        return list_remove, elem_add
    
    
class CustomRewardOnActivation(RewardOnActivation):
    def __init__(self, agent_name, out_reward, timelimit, coop=True, **kwargs):
        super().__init__(reward=0, **kwargs)
        self.agent_name = agent_name
        self.coop = coop
        self._reward = 0
        self.active = False
        self.spawned = False

        if out_reward != "no_object":
            self.out_reward = out_reward
        else:
            self.out_reward = None
        
        self.condition_satisfied = False
        self.timelimit = timelimit

    def activate(self, activating):
        list_remove = None
        elem_add = None
        is_activated = False

        if self.coop:
            if activating.name == self.agent_name or activating.name == self.partner_landmark.name:
                is_activated = True
        else:
            if isinstance(activating, BaseAgent) or activating.name == self.partner_landmark.name:
                is_activated = True
        
        if is_activated:
            if self.spawned is False:
                self.current_timelimit = self.timelimit
                self.active = True
                self._texture_surface.fill(color=(255, 100, 100))
                if self.partner_landmark.active:
                    self.condition_satisfied = True
                    self.partner_landmark.condition_satisfied = True
                    self._texture_surface.fill(color=(255, 255, 255))
                    self.partner_landmark._texture_surface.fill(color=(255, 255, 255))

                    self_spawn_coordinates, partner_spawn_coordinates = self.get_spawn_coordinates()
                    
                    if self.out_reward is not None:
                        elem_add = [(self.out_reward, self_spawn_coordinates)]
                    self.spawned = True

                    if self.partner_landmark.spawned is False:
                        self.partner_landmark.spawned = True
                        if self.partner_landmark.out_reward is not None:
                            if elem_add is not None:
                                elem_add.append((self.partner_landmark.out_reward, partner_spawn_coordinates))
                            else:
                                elem_add = [(self.partner_landmark.out_reward, partner_spawn_coordinates)]

        return list_remove, elem_add
    
    def check_if_active(self):
        if self.active:
            if self.current_timelimit > 0:
                self.activated = True
                self.current_timelimit -= 1
            else:
                self.active = False
                if self.spawned is False:
                    self._texture_surface.fill(color=(100, 200, 100))
                else:
                    self._texture_surface.fill(color=(255, 255, 255))
        else:
            self.active = False
    
    def add_partner(self, partner_landmark):
        self.partner_landmark = partner_landmark

    def get_spawn_coordinates(self):
        x, y = self.coordinates[0][0], self.coordinates[0][1]
        self_spawn_coordinates_center = self.adjust_coordinates(x, y)

        x, y = self.partner_landmark.coordinates[0][0], self.partner_landmark.coordinates[0][1]
        partner_spawn_coordinates_center = self.adjust_coordinates(x, y)

        self_spawn_coordinates_sampler = CoordinateSampler(self_spawn_coordinates_center, "rectangle", (30, 30))
        partner_spawn_coordinates_sampler = CoordinateSampler(partner_spawn_coordinates_center, "rectangle", (30, 30))
        self_spawn_coordinates = self_spawn_coordinates_sampler.sample()
        partner_spawn_coordinates = partner_spawn_coordinates_sampler.sample()
        return self_spawn_coordinates, partner_spawn_coordinates
    
    def adjust_coordinates(self, x, y):
        if round(x, -1) == 20 or round(x, -1) == 280:
            if round(x, -1) == 20:
                x = x + 50
            else:
                x = x - 50
        else:
            if round(y, -1) == 20:
                y = y + 50
            else:
                y = y - 50
        return (x, y)

class CustomRewardOnDoubleActivation(RewardOnActivation):
    def __init__(self, first_agent, out_reward, timelimit, sampled_stage, second_agent=None, coop=True, **kwargs):
        super().__init__(reward=0, **kwargs)
        self.first_agent = first_agent
        self.second_agent = second_agent
        self.sampled_stage = sampled_stage
        self.coop = coop
        self.out_reward = out_reward
        self._reward = 0
        self.active = False
        self.spawned = False
        
        self.condition_satisfied = False
        self.first_agent_activated = False
        self.second_agent_activated = False
        self.timelimit = timelimit
        self.agent_timelimits = [0, 0]

    def activate(self, activating):
        list_remove = None
        elem_add = None

        if self.spawned is False:
            if activating.name == self.first_agent:
                self.first_agent_activated = True
                self.agent_timelimits[0] = self.timelimit
                self._texture_surface.fill(color=(0, 100, 255))
            elif activating.name == self.second_agent:
                self.second_agent_activated = True
                self.agent_timelimits[1] = self.timelimit
                self._texture_surface.fill(color=(0, 100, 255))
            
            if self.coop:
                if self.second_agent is None:
                    if self.first_agent_activated:
                        self.spawned = True
                        self._texture_surface.fill(color=(255, 255, 255))
                        self_spawn_coordinates = self.get_spawn_coordinates()
                        elem_add = [(self.out_reward, self_spawn_coordinates)]
                        self.condition_satisfied = True

                else:
                    if self.first_agent_activated and self.second_agent_activated:
                        self.spawned = True
                        self._texture_surface.fill(color=(255, 255, 255))
                        self_spawn_coordinates = self.get_spawn_coordinates()
                        elem_add = [(self.out_reward, self_spawn_coordinates)]
                        self.condition_satisfied = True


            else:
                if self.first_agent_activated or self.second_agent_activated:
                    self.spawned = True
                    self._texture_surface.fill(color=(255, 255, 255))
                    self_spawn_coordinates = self.get_spawn_coordinates()
                    elem_add = [(self.out_reward, self_spawn_coordinates)]
                    self.condition_satisfied = True


        return list_remove, elem_add
    
    def check_if_active_and_stage(self, stage):
        for i in range(len(self.agent_timelimits)):
            if self.agent_timelimits[i] > 0:
                self.agent_timelimits[i] -= 1
            else:
                if i == 0:
                    
                    if self.condition_satisfied is False and self.first_agent_activated is True:
                        self._texture_surface.fill(color=(35, 200, 50))
                        self.first_agent_activated = False
                else:
                    if self.condition_satisfied is False and self.second_agent_activated is True:
                        self._texture_surface.fill(color=(35, 200, 50))
                        self.second_agent_activated = False
                        
        #Check which stage environment is in                
        self.stage = stage #Stage not used for now as a spawn restriction

        

    def get_spawn_coordinates(self):
        x, y = self.coordinates[0][0], self.coordinates[0][1]
        self_spawn_coordinates_center = self.adjust_coordinates(x, y)

        self_spawn_coordinates_sampler = CoordinateSampler(self_spawn_coordinates_center, "rectangle", (30, 30))
        self_spawn_coordinates = self_spawn_coordinates_sampler.sample()
        return self_spawn_coordinates
    
    def adjust_coordinates(self, x, y):
        if round(x, -1) == 20 or round(x, -1) == 280:
            if round(x, -1) == 20:
                x = x + 50
            else:
                x = x - 50
        else:
            if round(y, -1) == 20:
                y = y + 50
            else:
                y = y - 50
        return (x, y)

    
class LemonDispenser(ActivableElement):
    def __init__(self, out_reward, radius, physical_shape, texture, name, agent_name, coop=True, **kwargs):

        super().__init__(
            config_key=ElementTypes.CANDY,
            radius=radius,
            physical_shape=physical_shape,
            texture=texture,
            movable=True,
            graspable=True,
            traverseable=True,
            mass=1,
            name=name,
            **kwargs
        )

        self.graspable = True
        self.agent_name = agent_name
        self.out_reward = out_reward
        self.coop = coop
        self.condition_satisfied = False
    
    def activate(self, activating):
        list_remove = None
        elem_add = None
        is_activated = False

        if self.coop:
            if activating.name == self.agent_name:
                is_activated = True
        else:
            if isinstance(activating, BaseAgent):
                is_activated = True
        
        if is_activated:
            list_remove = [self]
            elem_add = [(self.out_reward, self.coordinates)]

        return list_remove, elem_add
    
    @property
    def terminate_upon_activation(self):
        return False


 

class Lemon(ContactElement):
    def __init__(self, radius, physical_shape, texture, name, agent_name, coop=True, **kwargs):

        super().__init__(
            config_key=ElementTypes.CANDY,
            radius=radius,
            physical_shape=physical_shape,
            texture=texture,
            movable=True,
            graspable=True,
            traverseable=True,
            mass=1,
            name=name,
            **kwargs
        )

        self.agent_name = agent_name
        self.coop = coop
        self.condition_satisfied = False
    
    def activate(self, activating):
        list_remove = None
        elem_add = None
        is_activated = False

        if self.coop:
            if activating.name == self.agent_name:
                is_activated = True
        else:
            if isinstance(activating, BaseAgent):
                is_activated = True
        
        if is_activated:
            self.condition_satisfied = True
            list_remove = [self]

        return list_remove, elem_add

    @property
    def terminate_upon_activation(self):
        return False
    
class InputOutputMachine(ActivableByGem):
    def __init__(self, radius, physical_shape, texture, name, reward, condition_obj=True, movable=False,
                  graspable=False, activation_zone=None, **kwargs):

        super().__init__(
            config_key=ElementTypes.CHEST,
            radius=radius,
            physical_shape=physical_shape,
            texture=texture,
            movable=movable,
            graspable=graspable,
            mass=20,
            name=name,
            **kwargs
        )

        self.graspable = True
        self.condition_obj = condition_obj
        self.reward = reward
        self.condition_satisfied = False
        self.activation_zone = activation_zone

    def _set_pm_collision_type(self):
        for pm_shape in self._pm_shapes:
            pm_shape.collision_type = CollisionTypes.ACTIVABLE_BY_GEM

    def activate(self, activating):
        
        list_remove = None
        elem_add = None

        if activating.elem_activated is self:
            if self.activation_zone is not None:
                if self.activation_zone.active:
                    self.condition_satisfied = True
                    list_remove = [activating]
                    elem_add = [(self.reward, self.coordinates)]
                else:
                    pass
            else:
                self.condition_satisfied = True
                if self.condition_obj:
                    coordinates = self.get_spawn_coordinates()
                    elem_add = [(self.reward, coordinates)]
                    list_remove = [activating]


        return list_remove, elem_add

    def get_spawn_coordinates(self):
        x, y = self.coordinates[0][0], self.coordinates[0][1]
        self_spawn_coordinates_center = self.adjust_coordinates(x, y)
        self_spawn_coordinates_sampler = CoordinateSampler(self_spawn_coordinates_center, "rectangle", (10, 10))
        self_spawn_coordinates = self_spawn_coordinates_sampler.sample()
        return self_spawn_coordinates
    
    def adjust_coordinates(self, x, y):
        if round(x, -1) == 20 or round(x, -1) == 280:
            if round(x, -1) == 20:
                x = x + 30
            else:
                x = x - 30
        else:
            if round(y, -1) == 20:
                y = y + 30
            else:
                y = y - 30
        return (x, y)

class TimedCustomRewardOnActivation(RewardOnActivation):
    def __init__(self, time_limit, **kwargs):
        super().__init__(reward=0, **kwargs)
        self._reward = 0
        self.active = False
        self.spawned = False
        
        self.condition_satisfied = False
        self.time_limit = time_limit
        self.activated = False

    def activate(self, activating):
        list_remove = None
        elem_add = None

        if isinstance(activating, BaseAgent):
            self.active_time_limit = self.time_limit
            self.active = True
            self._texture_surface.fill(color=(0, 100, 255))

        return list_remove, elem_add
    
    def check_if_active(self):
        if self.active:
            if self.active_time_limit > 0:
                self.activated = True
                self.active_time_limit -= 1
            else:
                self.active = False
                self._texture_surface.fill(color=(100, 200, 100))
        else:
            self.active = False

class ConditionActiveElement(RewardOnActivation):
    def __init__(self, out_reward, radius, physical_shape, texture, name, activation_zone, **kwargs):

        super().__init__(
            #config_key=ElementTypes.REWARD_ON_ACTIVATION,
            radius=radius,
            physical_shape=physical_shape,
            texture=texture,
            movable=True,
            graspable=True,
            traverseable=True,
            mass=1,
            reward=0,
            name=name,
            **kwargs
        )

        self.condition_satisfied = False
        self.activation_zone = activation_zone
        self.out_reward = out_reward
    
    def activate(self, activating):
        list_remove = None
        elem_add = None

        if isinstance(activating, BaseAgent):
            if self.activation_zone.active:
                self.condition_satisfied = True
                list_remove = [self]
                elem_add = [(self.out_reward, self.coordinates)]

            else:
                pass

        return list_remove, elem_add

    @property
    def terminate_upon_activation(self):
        return False

 



class CoopCraftingEnv(MultiAgentEnv):
    def __init__(self, config):
        super(CoopCraftingEnv, self).__init__()

        self.config = config
        self.num_goals = config["num_landmarks"]
        self.num_agents = config["num_agents"]
        self.timelimit = config["timelimit"]
        self.coop_chance = config["coop_chance"]
        self.forced_coop_rate = config["forced_coop_rate"]
        self.playground_height = config["playground_height"]
        self.playground_width = config["playground_width"]
        self.resolution = config["agent_resolution"]
        self.seed = config["seed"]
        self.stages = config["stages"]
        self.episodes = 0
        self.time_steps = 0
        self.truncated = False

        #Playgrounds for different agents in case of non coop. Only works for two agent case 
        self.shared_playground = GridRooms(size=(self.playground_height, self.playground_width), room_layout=(2, 2), 
                                    random_doorstep_position=False, doorstep_size=80)
        self.agent_0_playground = GridRooms(size=(self.playground_height, self.playground_width), room_layout=(2, 2),
                                    random_doorstep_position=False, doorstep_size=80)
        self.agent_1_playground = GridRooms(size=(self.playground_height, self.playground_width), room_layout=(2, 2),
                                    random_doorstep_position=False, doorstep_size=80)
        
        self.agent_ids = set()
        self.single_reward = config["single_reward"]

        if "test" in config.keys():
            self.test = True
        else:
            self.test = False
        
        if "agents_per_env" in config.keys():
            self.num_agents = config["agents_per_env"]

        
        
        self.shared_engine = Engine(
            playground=self.shared_playground, time_limit=(self.timelimit + 1),  
        )
        self.agent_0_engine = Engine(
            playground=self.agent_0_playground, time_limit=(self.timelimit + 1),  
        )
        self.agent_1_engine = Engine(
            playground=self.agent_1_playground, time_limit=(self.timelimit + 1),  
        )

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
        
        self.observation_space = spaces.Box(high=1, low=0, shape=(self.resolution, self.resolution, 3), dtype=np.float32)

        self.success_rate_dict = {}
        for s in range(1, self.stages + 1):
            self.success_rate_dict["stage_{0}".format(s)] = {"agent_0": [False], "agent_1": [False]}

        self.last_coop = False
        

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
                        actions[agent][actuator] = int(torch.round(self.clip_actions(act, act_idx)))
                        act_idx += 1
        
        if self.coop:
            #Handle timer for pressure plate
            for element in self.shared_playground.elements:
                if isinstance(element, (TimedCustomRewardOnActivation, CustomRewardOnActivation)):
                    element.check_if_active()
                elif isinstance(element, (CustomRewardOnDoubleActivation)):
                    element.check_if_active_and_stage(self.active_stage+1)

            self.shared_engine.step(actions)
            self.shared_engine.update_observations()
        else:
            for element in self.agent_0_playground.elements:
                if isinstance(element, (TimedCustomRewardOnActivation, CustomRewardOnActivation)):
                    element.check_if_active()
                elif isinstance(element, (CustomRewardOnDoubleActivation)):
                    element.check_if_active_and_stage(self.active_stage+1)

            agent_0_actions = {self._active_agents[0]: actions[self._active_agents[0]]}
            self.agent_0_engine.step(agent_0_actions)

            for element in self.agent_1_playground.elements:
                if isinstance(element, (TimedCustomRewardOnActivation, CustomRewardOnActivation)):
                    element.check_if_active()
                elif isinstance(element, (CustomRewardOnDoubleActivation)):
                    element.check_if_active_and_stage(self.active_stage+1)

            agent_1_actions = {self._active_agents[1]: actions[self._active_agents[1]]}
            self.agent_1_engine.step(agent_1_actions)



            self.agent_0_engine.update_observations()
            self.agent_1_engine.update_observations()

        observations = self.process_obs()
        if self.coop:
            rewards, dones, truncated, info = self.compute_reward_every_stage(self.stages, self.task_dict)
        else:
            rewards, dones, truncated, info = self.compute_reward_every_stage(self.stages, self.task_dict_agent_0, self.task_dict_agent_1)

        return observations, rewards, dones, truncated, info
    

    def reset(self, seed=None, options=None):
        if self.episodes > 0:
            if self.last_coop:
                for agent in self._active_agents:
                    self.shared_playground.remove_agent(agent)
                self.shared_engine.reset()
            
            else:
                self.agent_0_playground.remove_agent(self._active_agents[0])

                if self.num_agents == 2:
                    self.agent_1_playground.remove_agent(self._active_agents[1])

                self.agent_0_engine.reset()
                self.agent_1_engine.reset()
            
        self.end_condition_object_has_existed = False

        self.coop = np.random.uniform() < self.coop_chance
        self.last_coop = self.coop
        
        #Build element coordinates
        element_coordinates = self.build_coordinates()
        env_coordinates = self.build_env_coordinates()

        #Possible objects
        possible_objects = [("circle",[255,255,0]),("circle",[0,255,55]),("circle",[255,0,255]),
                        ("pentagon",[255,255,0]),("pentagon",[0,255,255]),("pentagon",[255,0,255]),
                        ("triangle",[255,255,0]),("triangle",[0,255,255]),("triangle",[255,0,255]),
                        ("circle",[255,255,0]),("circle",[0,255,55]),("circle",[255,0,255]),       
                        ("pentagon",[255,255,0]),("pentagon",[0,255,255]),("pentagon",[255,0,255]),
                        ("triangle",[255,255,0]),("triangle",[0,255,255]),("triangle",[255,0,255])] #Copied as a hack to allow for
                                                                                                    #more stages as objects run out
        
        end_conditions = ["no_object", "object_exists"]

        self.agent_goal_dict = {}
        self.agent_first_reward_dict = {}
        if self.coop:
            self.spawn_agents(element_coordinates, 2, self.shared_playground)
            #stage = self.stage_scheduler() #Change later but for now we only use 3 stages and sampling does not work for non coop
            forced_coop = np.random.uniform() < self.forced_coop_rate
            self.task_dict = self.sample_task_tree(self.stages, end_conditions, possible_objects, element_coordinates, 
                                              env_coordinates, 2, playground=self.shared_playground, forced_coop=forced_coop)
            self._active_agents = self.shared_playground.agents.copy()

        else:
            self.spawn_agents(element_coordinates, 1, self.agent_0_playground, "agent_0")
            self.spawn_agents(element_coordinates, 1, self.agent_1_playground, "agent_1")

            self.task_dict_agent_0 = self.sample_task_tree(self.stages, end_conditions, possible_objects.copy(), element_coordinates.copy(), 
                                              env_coordinates.copy(), 1, playground=self.agent_0_playground, agent_name="agent_0")
            
            self.task_dict_agent_1 = self.sample_task_tree(self.stages, end_conditions, possible_objects.copy(), element_coordinates.copy(),
                                                env_coordinates.copy(), 1, playground=self.agent_1_playground, agent_name="agent_1")
            self._active_agents = []
            self._active_agents.append(self.agent_0_playground.agents[0])
            
            if self.num_agents == 2:
                self._active_agents.append(self.agent_1_playground.agents[0]) 
            elif self.num_agents > 2:
                assert False, "Only one or two agents supported for now"
        
       
        info = {}
        self.stage_first_reward_dict = {}
        for agent in self._active_agents:
            self.stage_first_reward_dict[agent.name] = {"stage_{0}".format(s): True for s in range(1, self.stages+1)}
        
        for s in range(1, self.stages+1):
            self.success_rate_dict["stage_{0}".format(s)] = {"agent_0": [False], "agent_1": [False]}
        
        if self.coop:
            self.shared_engine.elapsed_time = 0
            self.shared_engine.update_observations()
        else:
            self.agent_0_engine.elapsed_time = 0
            self.agent_1_engine.elapsed_time = 0
            self.agent_0_engine.update_observations()
            self.agent_1_engine.update_observations()

        self.time_steps = 0
        self.active_stage = 0
        self.episodes += 1
        observations = self.process_obs()
        return observations, info
    
    def compute_reward_every_stage(self, stage, task_dict, second_task_dict=None):

        time_till_end = (self.timelimit - self.time_steps) / self.timelimit

        rewards = {}
        dones = {}
        truncateds = {}
        infos = {}
        
        for agent in self._active_agents:

            if second_task_dict is not None:
                if agent.name == "agent_0":
                    task_dict = task_dict
                else:
                    task_dict = second_task_dict
            
            task_successes = {"activate_landmarks":0,
                                   "double_activate":0,
                                   "lemon_hunt":0,
                                   "crafting":0,
                                   "in_out_machine":0,
                                   "dropoff":0,
                                   "pressure_plate":0}
        
            tasks_sampled =  {"activate_landmarks":0,
                                   "double_activate":0,
                                   "lemon_hunt":0,
                                   "crafting":0,
                                   "in_out_machine":0,
                                   "dropoff":0,
                                   "pressure_plate":0}
        
            for stage in range(1, stage+1):
                stage_task = task_dict["stage_{0}".format(stage)]["task"]
                tasks_sampled[stage_task] += 1
         
            for s in range(1, stage+1):
                condition_obj = task_dict["stage_{0}".format(s)]["condition_object"]
                if condition_obj.condition_satisfied:
                    #Log which task was succesfully completed
                    task = task_dict["stage_{0}".format(s)]["task"]
                    task_successes[task] += 1

                    idx = 0
                    switch = False #switch to compute success rates correctly for logging. Keeping the old version for consistency during training
                    if switch:
                        if self.success_rate_dict["stage_{0}".format(s)][agent.name][-1] == True:
                            while int(s+idx) < stage and self.success_rate_dict["stage_{0}".format(int(s+idx))][agent.name][-1] == True:
                                idx += 1
                            self.success_rate_dict["stage_{0}".format(int(s+idx))][agent.name].append(True)
                        else:
                            while int(s-idx) > 1 and self.success_rate_dict["stage_{0}".format(int(s-idx-1))][agent.name][-1] == False:
                                idx += 1
                            self.success_rate_dict["stage_{0}".format(int(s-idx))][agent.name].append(True)
                    else:
                        self.success_rate_dict["stage_{0}".format(s)][agent.name].append(True)
                    
                else:
                    self.success_rate_dict["stage_{0}".format(s)][agent.name].append(False)
        
            stage_success = []
            for s in range(1, stage + 1):
                stage_success.append(any(self.success_rate_dict["stage_{0}".format(s)][agent.name]))
            
            reward = sum(stage_success)
            self.active_stage = reward #Which stage the environment is in right now
            

                
            if self.stage_first_reward_dict[agent.name]["stage_{0}".format(stage)] and int(reward) == stage:
                infos[agent.name] = {"success": 1.0, "goal_line": 0.0, "true_goal":  self.agent_goal_dict[agent.name], "time_till_end": time_till_end,
                                     "tasks_sampled": tasks_sampled, "task_successes": task_successes}
            else:
                infos[agent.name] = {"success": 0.0, "goal_line": 0.0, "true_goal":  self.agent_goal_dict[agent.name], "time_till_end": time_till_end,
                                     "tasks_sampled": tasks_sampled, "task_successes": task_successes}
            
            for s in range(1, stage + 1):
                if self.success_rate_dict["stage_{0}".format(s)][agent.name][-1]: 
                    if self.stage_first_reward_dict[agent.name]["stage_{0}".format(s)]:

                        infos[agent.name]["success_stage_{0}".format(s)] = 1.0

                        if self.coop:
                            infos[agent.name]["coop_success_stage_{0}".format(s)] = 1.0
                        else:
                            infos[agent.name]["single_success_stage_{0}".format(s)] = 1.0

                        self.stage_first_reward_dict[agent.name]["stage_{0}".format(s)] = False

                    else:
                        infos[agent.name]["success_stage_{0}".format(s)] = 0.0

                        if self.coop:
                            infos[agent.name]["coop_success_stage_{0}".format(s)] = 0.0
                        else:
                            infos[agent.name]["single_success_stage_{0}".format(s)] = 0.0
                else:
                    infos[agent.name]["success_stage_{0}".format(s)] = 0.0

                    if self.coop:
                        infos[agent.name]["coop_success_stage_{0}".format(s)] = 0.0
                    else:
                        infos[agent.name]["single_success_stage_{0}".format(s)] = 0.0
                    
            for s in range(stage+1, self.stages+1):
                infos[agent.name]["success_stage_{0}".format(s)] = -1.0

                if self.coop:
                    infos[agent.name]["coop_success_stage_{0}".format(s)] = -1.0
                else:
                    infos[agent.name]["single_success_stage_{0}".format(s)] = -1.0

            for s in range(1, self.stages + 1):
                
                if self.coop:
                    infos[agent.name]["single_success_stage_{0}".format(s)] = -1.0
                
                else:
                    infos[agent.name]["coop_success_stage_{0}".format(s)] = -1.0
    
            rewards[agent.name] = 0.1 * reward ** 2
            #rewards[agent.name] = 0.0

            if second_task_dict is not None:
                if agent.name == "agent_0":
                    done = self.agent_0_playground.done or not self.agent_0_engine.game_on
                    truncated = self.agent_0_playground.done or not self.agent_0_engine.game_on
                else:
                    done = self.agent_1_playground.done or not self.agent_1_engine.game_on
                    truncated = self.agent_1_playground.done or not self.agent_1_engine.game_on
            else:
                done = self.shared_playground.done or not self.shared_engine.game_on
                truncated = self.shared_playground.done or not self.shared_engine.game_on

            dones[agent.name] = done
            truncateds[agent.name] = truncated

        dones["__all__"] = all(dones.values())
        truncateds["__all__"] = all(truncateds.values())
        return rewards, dones, truncateds, infos
    
    

    def spawn_agents(self, element_coordinates, num_agents, playground, agent_name=None):
        if agent_name is not None:
            assert num_agents == 1, "Only one agent allowed for now"

        sample_pos_agents = random.sample(element_coordinates, num_agents)
        possible_agent_samplers = []
        for i in range(num_agents):
            element_coordinates.remove(sample_pos_agents[i])
            agent_sampler = CoordinateSampler(sample_pos_agents[i], area_shape="rectangle", size=(20, 40))
            possible_agent_samplers.append(agent_sampler)

        
        possible_agent_colors = [(255, 255, 255), (170, 170, 170), (0, 0, 255)]
        agent_ls = []
        for i in range(num_agents):
            if agent_name is not None:
                if agent_name == "agent_0":
                    color = (255, 255, 255)
                elif agent_name == "agent_1":
                    color = (170, 170, 170)
                else:
                    assert False, "Agent name not recognized"
            else:
                color = possible_agent_colors[i]
            agent = BaseAgent(
            controller=External(),
            radius=12,
            mass=15,
            interactive=True, 
            name="agent_{0}".format(i) if agent_name is None else agent_name,
            texture=UniqueCenteredStripeTexture(size=10,
                color=color, color_stripe=(0,0,0), size_stripe=4),
            temporary=True)
            #Makes agents traversable
            categories = 2**3
            for p in agent.parts:
                p.pm_visible_shape.filter = pymunk.ShapeFilter(categories)
            name = "agent_{0}".format(i) if agent_name is None else agent_name
            self.agent_goal_dict[name] = np.zeros(self.num_goals, dtype=int)
            self.agent_ids.add(name)
            agent_ls.append(agent)
      
        for agent, idx in zip(agent_ls, range(self.num_agents)):
            ignore_agents = [agent_ig.parts for agent_ig in agent_ls if agent_ig != agent]
            ignore_agents = [agent_part for agent_ls in ignore_agents for agent_part in agent_ls]
            agent.add_sensor(TopdownSensor(agent.base_platform, fov=360, resolution=self.resolution, max_range=160, normalize=True))
            playground.add_agent(agent, possible_agent_samplers[idx], allow_overlapping=True, max_attempts=10)
    
    
    def sample_task_tree(self, num_stages, end_conditions, possible_objects, element_coordinates, 
                         env_coordinates, num_agents, num_distractors=0, playground=None, agent_name=None, forced_coop=True):

        possible_object_types = possible_objects.copy()
        task_dict = {}
        end_condition = random.choice(end_conditions)
        end_condition_object = random.choice(possible_object_types)
        possible_object_types.remove(end_condition_object)
        end_condition_object_shape = end_condition_object[0]
        end_condition_object_color = end_condition_object[1]

        if end_condition == "object_exists":
            end_condition_object = Chest(physical_shape=end_condition_object_shape, radius=10, 
                                       texture=ColorTexture(color=end_condition_object_color, size=10), 
                                        name="end_condition_object", condition_obj=True, temporary=True)
        else:
            end_condition_object = "no_object"
        

        task_dict["end_condition"] = end_condition
        task_dict["end_condition_object"] = end_condition_object

        task_out_objects = [end_condition_object]
        needed_env_objects = []
        assigned_stage_tasks = []
        for s in range(num_stages, 0, -1):
            stage_task_type = self.sample_stage_task(s, num_stages, end_condition, assigned_stage_tasks)
            assigned_stage_tasks.append(stage_task_type)
            needed_in_objects, needed_env_object, condition_obj = self.task_creator(stage_task_type, task_out_objects, possible_object_types, 
                                                                                    s, num_agents, agent_name=agent_name, forced_coop=forced_coop)
            needed_env_objects.append(needed_env_object)
            task_dict["stage_{0}".format(s)] = {}
            task_dict["stage_{0}".format(s)]["task"] = stage_task_type
            task_dict["stage_{0}".format(s)]["in_objects"] = [obj.name for obj in needed_in_objects]
            task_dict["stage_{0}".format(s)]["out_object"] = [obj.name for obj in task_out_objects if obj != "no_object"]
            task_dict["stage_{0}".format(s)]["condition_object"] = condition_obj
            task_out_objects = needed_in_objects
        
        task_dict["num_stages"] = num_stages
        
        random.shuffle(element_coordinates)
        for object, c in zip(task_out_objects, range(len(task_out_objects))):
            if object != "no_object":
                object_coordinates = CoordinateSampler(element_coordinates[c], area_shape="rectangle", size=(10, 10))

                if playground is None:
                    self.playground.add_element(object, object_coordinates)
                else:
                    playground.add_element(object, object_coordinates)

        random.shuffle(env_coordinates)
        needed_env_objects = [obj for sublist in needed_env_objects for obj in sublist]
        for env_obj, c in zip(needed_env_objects, range(len(needed_env_objects))):
            env_object_coordinates = CoordinateSampler(env_coordinates[c], area_shape="rectangle", size=(10, 10))

            if playground is None:
                self.playground.add_element(env_obj, env_object_coordinates)
            else:
                playground.add_element(env_obj, env_object_coordinates)

        if num_distractors > 0:
            sampled_num_distractors = random.randint(0, num_distractors)
        else:
            sampled_num_distractors = 0
        for d in range(sampled_num_distractors):
            distractor_object = random.choice(possible_object_types)
            possible_object_types.remove(distractor_object)
            distractor_object_shape = distractor_object[0]
            distractor_object_color = distractor_object[1]

            distractor_object = Chest(physical_shape=distractor_object_shape, radius=10, 
                                        texture=ColorTexture(color=distractor_object_color, size=10), 
                                        name="distractor_object_{0}".format(d), condition_obj=False, temporary=True)
            distractor_object_coordinates = CoordinateSampler(element_coordinates[d], area_shape="rectangle", size=(10, 10))

            if playground is None:
                self.playground.add_element(distractor_object, distractor_object_coordinates)
            else:
                playground.add_element(distractor_object, distractor_object_coordinates)

        return task_dict
    
    def task_creator(self, stage_task_type, task_out_objects, possible_object_types, stage, num_agents, agent_name=None, forced_coop=True):
        needed_in_objects = []
        needed_env_object = []
        if stage_task_type == "crafting":
            for object, n in zip(task_out_objects, range(len(task_out_objects))):
                if n == 0:
                    object_type = random.choice(possible_object_types)
                    possible_object_types.remove(object_type)
                    object_shape = object_type[0]
                    object_color = object_type[1]
                    

                    chest_object = Chest(physical_shape=object_shape, 
                                        radius=10, 
                                        texture=ColorTexture(color=object_color, size=10),
                                        out_reward=object, 
                                        name="chest_object_{0}".format(stage),
                                        temporary=True)
                    
                    object_type = random.choice(possible_object_types)
                    possible_object_types.remove(object_type)
                    object_shape = object_type[0]
                    object_color = object_type[1]
                    

                    diamond_object = Diamond(chest_object, physical_shape=object_shape, 
                                            radius=10, 
                                            texture=ColorTexture(color=object_color, size=10), 
                                            name="diamond_object_{0}".format(stage),
                                            temporary=True)
                    
                    needed_in_objects.append(diamond_object)
                    needed_in_objects.append(chest_object)

                    condition_obj = chest_object
                else:
                    needed_in_objects.append(object)
                
        elif stage_task_type == "dropoff":
            assert task_out_objects[0] == "no_object"

            dropoff = Chest(physical_shape="rectangle", 
                            radius=15,
                            texture=ColorTexture(color=[100, 100, 200], size=15),
                            condition_obj=False, movable=False, graspable=False,
                            dropoff=True,
                            name="dropoff",
                            temporary=True)
            
            object = random.choice(possible_object_types)
            possible_object_types.remove(object)
            object_shape = object[0]
            object_color = object[1]

            dropoff_diamond = Diamond(dropoff, 
                                      physical_shape=object_shape, 
                                      radius=10,
                                      texture=ColorTexture(color=object_color, size=10),
                                      name="dropoff_diamond_{0}".format(stage),
                                      temporary=True)
            
            needed_in_objects.append(dropoff_diamond)
            needed_env_object.append(dropoff)
            condition_obj = dropoff
        
        elif stage_task_type == "activate_landmarks":
            possible_agent_names = []
            if num_agents < 2:
                time_limit = 300
                assert agent_name is not None
                for i in range(2):
                    possible_agent_names.append(agent_name)
            else:
                if forced_coop:
                    time_limit = 2
                else:
                    time_limit = 300

                for i in range(2):
                    possible_agent_names.append("agent_{0}".format(i))
            
            first_agent = random.choice(possible_agent_names)
            possible_agent_names.remove(first_agent)
            second_agent = possible_agent_names[0]


            landmark1 = CustomRewardOnActivation(agent_name=first_agent, 
                                                radius=15,
                                                physical_shape="rectangle",
                                                texture=ColorTexture(color=[100, 200, 100], size=15),
                                                out_reward=task_out_objects[0],
                                                name="landmark0",
                                                timelimit=time_limit,
                                                coop=forced_coop, 
                                                temporary=True)

            landmark2 = CustomRewardOnActivation(agent_name=second_agent, 
                                                radius=15,
                                                physical_shape="rectangle",
                                                texture=ColorTexture(color=[100, 200, 100], size=15),
                                                out_reward=task_out_objects[1] if len(task_out_objects) > 1 else None,
                                                name="landmark1",
                                                timelimit=time_limit,
                                                coop=forced_coop,
                                                temporary=True)
            
            landmark1.add_partner(landmark2)
            landmark2.add_partner(landmark1)
            
            if len (task_out_objects) > 2:
                needed_in_objects = [obj for obj in task_out_objects[2:]]
            else:
                needed_in_objects = []

            needed_env_object.append(landmark1)
            needed_env_object.append(landmark2)
            condition_obj = landmark1
    
        elif stage_task_type == "lemon_hunt":
            possible_agent_names = []
            if num_agents < 2:
                assert agent_name is not None
                for i in range(2):
                    possible_agent_names.append(agent_name)
            else:
                for a in range(num_agents):
                    possible_agent_names.append("agent_{0}".format(a))

            lemon_agent = random.choice(possible_agent_names)
            possible_agent_names.remove(lemon_agent)
            agent_name = possible_agent_names[0]

            object = random.choice(possible_object_types)
            possible_object_types.remove(object)
            object_shape = object[0]
            object_color = object[1]

            lemon = Lemon(physical_shape=object_shape, 
                        radius=10,
                        texture=object_color,
                        name="lemon_{0}".format(stage), 
                        agent_name=lemon_agent,
                        coop=forced_coop,
                        temporary=True)
            
            object = random.choice(possible_object_types)
            possible_object_types.remove(object)
            object_shape = object[0]
            object_color = object[1]

            lemon_dispenser = LemonDispenser(agent_name=agent_name, 
                                            radius=10,
                                            texture=ColorTexture(color=object_color, size=15),
                                            physical_shape=object_shape,
                                            out_reward=lemon,
                                            name="lemon_dispenser_{0}".format(stage),
                                            coop=forced_coop,
                                            temporary=True)
            
            needed_in_objects.append(lemon_dispenser)
            condition_obj = lemon
        
        elif stage_task_type == "pressure_plate":
            assert task_out_objects[0] != "no_object"
            
            if num_agents < 2:
                time_limit = 200
            else:
                time_limit = 2

            pressure_plate =  TimedCustomRewardOnActivation(radius=15, 
                                                            time_limit=time_limit, 
                                                            physical_shape="rectangle",
                                                            texture=ColorTexture(color=[100, 200, 100], size=15),
                                                            name="pressure_plate",
                                                            temporary=True)
            
            object = random.choice(possible_object_types)
            possible_object_types.remove(object)
            object_shape = object[0]
            object_color = object[1]

            pressure_plate_in_out = InputOutputMachine(physical_shape="rectangle", 
                                                        radius=15,
                                                        texture=ColorTexture(color=[50, 50, 200], size=15),
                                                        condition_obj=True,
                                                        activation_zone=pressure_plate,
                                                        name="in_out_machine_pressure_plate", 
                                                        reward=task_out_objects[0],
                                                        temporary=True)
            
            pressure_plate_diamond = Diamond(pressure_plate_in_out, 
                                             physical_shape=object_shape, 
                                             radius=10,
                                             texture=ColorTexture(color=object_color, size=10),
                                             name="pressure_plate_diamond_{0}".format(stage),
                                             temporary=True)

            needed_in_objects.append(pressure_plate_diamond)
            for obj in task_out_objects[1:]:
                if obj != "no_object":
                    needed_in_objects.append(obj)  

            needed_env_object.append(pressure_plate)
            needed_env_object.append(pressure_plate_in_out)
            condition_obj = pressure_plate_in_out

        elif stage_task_type == "double_activate":
            assert task_out_objects[0] != "no_object"
            possible_agent_names = []

            if num_agents < 2:
                assert agent_name is not None
                for i in range(2):
                    possible_agent_names.append(agent_name)
            else:
                for i in range(num_agents):
                    possible_agent_names.append("agent_{0}".format(i))

            
            double_activate_landmark = CustomRewardOnDoubleActivation(first_agent=possible_agent_names[0], 
                                                radius=15,
                                                sampled_stage=stage,
                                                physical_shape="rectangle",
                                                texture=ColorTexture(color=[35, 200, 50], size=15),
                                                out_reward=task_out_objects[0],
                                                name="landmark0",
                                                timelimit=10,
                                                coop=forced_coop,
                                                second_agent=None if num_agents < 2 else possible_agent_names[1],
                                                temporary=True)
            
            if len (task_out_objects) > 1:
                needed_in_objects = [obj for obj in task_out_objects[1:]]
            else:
                needed_in_objects = []

            needed_env_object.append(double_activate_landmark)
            condition_obj = double_activate_landmark

        
        else:
            assert task_out_objects[0] != "no_object"
            in_out_machine = InputOutputMachine(physical_shape="rectangle", 
                                                radius=15,
                                                texture=ColorTexture(color=[50, 50, 200], size=15),
                                                condition_obj=True,
                                                name="in_out_machine", reward=task_out_objects[0],
                                                temporary=True)
            
            object = random.choice(possible_object_types)
            possible_object_types.remove(object)
            object_shape = object[0]
            object_color = object[1]
        
            in_out_machine_diamond = Diamond(in_out_machine, 
                                             physical_shape=object_shape, 
                                             radius=10,
                                             texture=ColorTexture(color=object_color, size=10),
                                             name="in_out_machine_diamond_{0}".format(stage),
                                             temporary=True)
            
            needed_in_objects.append(in_out_machine_diamond)
            for obj in task_out_objects[1:]:
                if obj != "no_object":
                    needed_in_objects.append(obj)

            needed_env_object.append(in_out_machine)
            condition_obj = in_out_machine                                            

        return needed_in_objects, needed_env_object, condition_obj
    
    def sample_stage_task(self, stage, num_stages, end_condition, assigned_stage_tasks):

        if stage == 1 and num_stages > 1:
            if "lemon_hunt" not in assigned_stage_tasks and "double_activate" not in assigned_stage_tasks:
                stage_task = random.choice(["activate_landmarks", "double_activate"])
            elif "lemon_hunt" not in assigned_stage_tasks and "double_activate" in assigned_stage_tasks:
                if "in_out_machine" not in assigned_stage_tasks:
                    stage_task = random.choice(["activate_landmarks", "crafting", "in_out_machine"])
                else:
                    stage_task = random.choice(["activate_landmarks", "crafting"])
            elif "lemon_hunt" in assigned_stage_tasks and "double_activate" not in assigned_stage_tasks:
                if "in_out_machine" not in assigned_stage_tasks:
                    stage_task = random.choice(["activate_landmarks", "double_activate", "in_out_machine", "crafting"])
                else:
                    stage_task = random.choice(["activate_landmarks", "double_activate", "crafting"])
               
           
        elif stage == num_stages and num_stages > 1:
            if end_condition == "no_object":
                stage_task = random.choice(["lemon_hunt", "dropoff", "crafting"])
            else:
                stage_task = random.choice(["crafting", "in_out_machine"])

        elif stage == num_stages and num_stages == 1:
            if end_condition == "no_object":
                stage_task = random.choice(["activate_landmarks", "lemon_hunt"])
            else:
                stage_task = random.choice(["activate_landmarks", "double_activate"])

        else:
            if "crafting" in assigned_stage_tasks:
                if "lemon_hunt" not in assigned_stage_tasks:
                    if "double_activate" not in assigned_stage_tasks:
                        if "in_out_machine" not in assigned_stage_tasks:
                            stage_task = random.choice(["double_activate", "in_out_machine"])
                        else:
                            stage_task = "double_activate"
                    else:
                        if "in_out_machine" not in assigned_stage_tasks:
                            stage_task = random.choice(["crafting", "in_out_machine"])
                        else:
                            stage_task = "crafting"
                else:
                    if "in_out_machine" not in assigned_stage_tasks:
                        stage_task = random.choice(["crafting", "in_out_machine"])
                    else:
                        stage_task = "crafting"
            else:
                if "in_out_machine" not in assigned_stage_tasks:
                    stage_task = random.choice(["crafting", "in_out_machine"])
                else:
                    stage_task = "crafting"

        return stage_task
    
    def build_coordinates(self):
        element_coordinates = []
        for w in range(50, self.playground_width, 50):
            x_coord = w
            for h in range(50, self.playground_height, 50):
                y_coord = h
                element_coordinates.append((x_coord, y_coord))
        element_coordinates.remove((self.playground_width // 2, self.playground_height // 2))
        return element_coordinates

    def build_env_coordinates(self):
        env_coordinates = []
        quarter_width = self.playground_width // 4
        quarter_height = self.playground_height // 4

        for i in [1, 3]:
            x_coord = i * quarter_width
            y_coord = i * quarter_height

            env_coordinates.append((x_coord, 20))
            env_coordinates.append((x_coord, int(self.playground_width-20)))

            env_coordinates.append((20, y_coord))
            env_coordinates.append((int(self.playground_height-20), y_coord))
        return env_coordinates
    
    def clip_actions(self, actions, act_idx):
        return np.clip(actions, self.action_space.low[act_idx], self.action_space.high[act_idx])
    
    def render(self):
        if self.coop:
            image = self.shared_engine.generate_agent_image(self.shared_playground.agents[0], max_size_pg=max(self.playground_height, self.playground_width))
        else:
            image = self.agent_0_engine.generate_agent_image(self.agent_0_playground.agents[0], max_size_pg=max(self.playground_height, self.playground_width))
        return image
       
    def close(self):
        self.engine.terminate()


class CoopCraftingEnvComm(MultiAgentEnv):
    def __init__(self, config):
        super(CoopCraftingEnvComm, self).__init__()

        self.config = config
        self.num_goals = config["num_landmarks"]
        self.num_agents = config["num_agents"]
        self.timelimit = config["timelimit"]
        self.coop_chance = config["coop_chance"]
        self.playground_height = config["playground_height"]
        self.playground_width = config["playground_width"]
        self.resolution = config["agent_resolution"]
        self.seed = config["seed"]
        self.stages = config["stages"]
        self.message_len = config["message_length"]
        self.vocab_size = config["vocab_size"]
        self.episodes = 0
        self.time_steps = 0
        self.truncated = False

        #Playgrounds for different agents in case of non coop. Only works for two agent case 
        self.shared_playground = GridRooms(size=(self.playground_height, self.playground_width), room_layout=(2, 2), 
                                    random_doorstep_position=False, doorstep_size=80)
        self.agent_0_playground = GridRooms(size=(self.playground_height, self.playground_width), room_layout=(2, 2),
                                    random_doorstep_position=False, doorstep_size=80)
        self.agent_1_playground = GridRooms(size=(self.playground_height, self.playground_width), room_layout=(2, 2),
                                    random_doorstep_position=False, doorstep_size=80)
        
        self.agent_ids = set()
        self.single_reward = config["single_reward"]

        if "test" in config.keys():
            self.test = True
        else:
            self.test = False
        
        if "agents_per_env" in config.keys():
            self.num_agents = config["agents_per_env"]

        
        
        self.shared_engine = Engine(
            playground=self.shared_playground, time_limit=(self.timelimit + 1),  
        )
        self.agent_0_engine = Engine(
            playground=self.agent_0_playground, time_limit=(self.timelimit + 1),  
        )
        self.agent_1_engine = Engine(
            playground=self.agent_1_playground, time_limit=(self.timelimit + 1),  
        )

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
        
        visual_observation_space = spaces.Box(high=1, low=0, shape=(self.resolution, self.resolution, 3), dtype=np.float32)
        message_observation_space = spaces.MultiDiscrete([self.vocab_size + 1 for m in range(self.message_len)])
        self.observation_space = spaces.Dict({
            "visual_observation_space": visual_observation_space,
            "message_observation_space": message_observation_space
        })
        

        self.success_rate_dict = {}
        for s in range(1, self.stages + 1):
            self.success_rate_dict["stage_{0}".format(s)] = {"agent_0": [False], "agent_1": [False]}

        self.last_coop = False

    def process_obs(self, messages):
        obs = {}
        id = self.num_agents - 1
        if self.coop:
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
        if action_dict:
            for agent in self._active_agents:
                messages[agent.name] = action_dict.get(agent.name)["message_action_space"]
                #info[agent.name] = {"message": messages[agent.name],
                #                  "goal": self.agent_goal_dict[agent.name]}
                agent_action = action_dict.get(agent.name)["actuators_action_space"]
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
        
        if self.coop:
            #Handle timer for pressure plate
            for element in self.shared_playground.elements:
                if isinstance(element, TimedCustomRewardOnActivation):
                    element.check_if_active()

            self.shared_engine.step(actions)
            self.shared_engine.update_observations()

        else:
            for element in self.agent_0_playground.elements:
                if isinstance(element, TimedCustomRewardOnActivation):
                    element.check_if_active()

            agent_0_actions = {self._active_agents[0]: actions[self._active_agents[0]]}
            self.agent_0_engine.step(agent_0_actions)

            for element in self.agent_1_playground.elements:
                if isinstance(element, TimedCustomRewardOnActivation):
                    element.check_if_active()

            agent_1_actions = {self._active_agents[1]: actions[self._active_agents[1]]}
            self.agent_1_engine.step(agent_1_actions)

            self.agent_0_engine.update_observations()
            self.agent_1_engine.update_observations()

        observations = self.process_obs(messages)
        if self.coop:
            rewards, dones, truncated, info = self.compute_reward_every_stage(self.stages, self.task_dict)
        else:
            rewards, dones, truncated, info = self.compute_reward_every_stage(self.stages, self.task_dict_agent_0, self.task_dict_agent_1)

        return observations, rewards, dones, truncated, info
    

    def reset(self, seed=None, options=None):
        if self.episodes > 0:
            if self.last_coop:
                for agent in self._active_agents:
                    self.shared_playground.remove_agent(agent)
                self.shared_engine.reset()
            
            else:
                self.agent_0_playground.remove_agent(self._active_agents[0])
                self.agent_1_playground.remove_agent(self._active_agents[1])

                self.agent_0_engine.reset()
                self.agent_1_engine.reset()
            
        self.end_condition_object_has_existed = False

        self.coop = np.random.uniform() < self.coop_chance
        self.last_coop = self.coop
        
        #Build element coordinates
        element_coordinates = self.build_coordinates()
        env_coordinates = self.build_env_coordinates()

        #Possible objects
        possible_objects = [("circle",[255,255,0]),("circle",[0,255,55]),("circle",[255,0,255]),
                        ("pentagon",[255,255,0]),("pentagon",[0,255,255]),("pentagon",[255,0,255]),
                        ("triangle",[255,255,0]),("triangle",[0,255,255]),("triangle",[255,0,255]),
                        ("circle",[255,255,0]),("circle",[0,255,55]),("circle",[255,0,255]),       
                        ("pentagon",[255,255,0]),("pentagon",[0,255,255]),("pentagon",[255,0,255]),
                        ("triangle",[255,255,0]),("triangle",[0,255,255]),("triangle",[255,0,255])] #Copied as a hack to allow for
                                                                                                    #more stages as objects run out
        
        end_conditions = ["no_object", "object_exists"]

        self.agent_goal_dict = {}
        self.agent_first_reward_dict = {}
        if self.coop:
            self.spawn_agents(element_coordinates, 2, self.shared_playground)
            #stage = self.stage_scheduler() #Change later but for now we only use 3 stages and sampling does not work for non coop
            self.task_dict = self.sample_task_tree(self.stages, end_conditions, possible_objects, element_coordinates, 
                                              env_coordinates, 2, playground=self.shared_playground)
            self._active_agents = self.shared_playground.agents.copy()

        else:
            self.spawn_agents(element_coordinates, 1, self.agent_0_playground, "agent_0")
            self.spawn_agents(element_coordinates, 1, self.agent_1_playground, "agent_1")

            #stage_agent_0 = self.stage_scheduler()
            self.task_dict_agent_0 = self.sample_task_tree(self.stages, end_conditions, possible_objects.copy(), element_coordinates.copy(), 
                                              env_coordinates.copy(), 1, playground=self.agent_0_playground, agent_name="agent_0")
            
            #stage_agent_1 = self.stage_scheduler()
            self.task_dict_agent_1 = self.sample_task_tree(self.stages, end_conditions, possible_objects.copy(), element_coordinates.copy(),
                                                env_coordinates.copy(), 1, playground=self.agent_1_playground, agent_name="agent_1")
            self._active_agents = []
            self._active_agents.append(self.agent_0_playground.agents[0])
            self._active_agents.append(self.agent_1_playground.agents[0])
        
       
        info = {}
        self.stage_first_reward_dict = {}
        init_messages = {}
        for agent in self._active_agents:
            self.stage_first_reward_dict[agent.name] = {"stage_{0}".format(s): True for s in range(1, self.stages+1)}
            init_messages[agent.name] = np.zeros(self.message_len, dtype=int)
        
        for s in range(1, self.stages+1):
            self.success_rate_dict["stage_{0}".format(s)] = {"agent_0": [False], "agent_1": [False]}
        
        if self.coop:
            self.shared_engine.elapsed_time = 0
            self.shared_engine.update_observations()
        else:
            self.agent_0_engine.elapsed_time = 0
            self.agent_1_engine.elapsed_time = 0
            self.agent_0_engine.update_observations()
            self.agent_1_engine.update_observations()

        self.time_steps = 0
        self.episodes += 1
        observations = self.process_obs(init_messages)
        return observations, info
    
    def compute_reward_every_stage(self, stage, task_dict, second_task_dict=None):

        time_till_end = (self.timelimit - self.time_steps) / self.timelimit

        rewards = {}
        dones = {}
        truncateds = {}
        infos = {}
        
        for agent in self._active_agents:

            if second_task_dict is not None:
                if agent.name == "agent_0":
                    task_dict = task_dict
                else:
                    task_dict = second_task_dict
         
            for s in range(1, stage+1):
                condition_obj = task_dict["stage_{0}".format(s)]["condition_object"]
                if condition_obj.condition_satisfied:
                    self.success_rate_dict["stage_{0}".format(s)][agent.name].append(True)
                else:
                    self.success_rate_dict["stage_{0}".format(s)][agent.name].append(False)
        
            stage_success = []
            for s in range(1, stage + 1):
                stage_success.append(any(self.success_rate_dict["stage_{0}".format(s)][agent.name]))
            
            reward = sum(stage_success)
            

                
            if self.stage_first_reward_dict[agent.name]["stage_{0}".format(stage)] and int(reward) == stage:
                infos[agent.name] = {"success": 1.0, "goal_line": 0.0, "true_goal":  self.agent_goal_dict[agent.name], "time_till_end": time_till_end}
            else:
                infos[agent.name] = {"success": 0.0, "goal_line": 0.0, "true_goal":  self.agent_goal_dict[agent.name], "time_till_end": time_till_end}
            
            for s in range(1, stage + 1):
                if self.success_rate_dict["stage_{0}".format(s)][agent.name][-1]: 
                    if self.stage_first_reward_dict[agent.name]["stage_{0}".format(s)]:

                        infos[agent.name]["success_stage_{0}".format(s)] = 1.0

                        if self.coop:
                            infos[agent.name]["coop_success_stage_{0}".format(s)] = 1.0
                        else:
                            infos[agent.name]["single_success_stage_{0}".format(s)] = 1.0

                        self.stage_first_reward_dict[agent.name]["stage_{0}".format(s)] = False
                    else:
                        infos[agent.name]["success_stage_{0}".format(s)] = 0.0

                        if self.coop:
                            infos[agent.name]["coop_success_stage_{0}".format(s)] = 0.0
                        else:
                            infos[agent.name]["single_success_stage_{0}".format(s)] = 0.0
                else:
                    infos[agent.name]["success_stage_{0}".format(s)] = 0.0

                    if self.coop:
                        infos[agent.name]["coop_success_stage_{0}".format(s)] = 0.0
                    else:
                        infos[agent.name]["single_success_stage_{0}".format(s)] = 0.0
                    
            for s in range(stage+1, self.stages+1):
                infos[agent.name]["success_stage_{0}".format(s)] = -1.0

                if self.coop:
                    infos[agent.name]["coop_success_stage_{0}".format(s)] = -1.0
                else:
                    infos[agent.name]["single_success_stage_{0}".format(s)] = -1.0

            for s in range(1, self.stages + 1):
                
                if self.coop:
                    infos[agent.name]["single_success_stage_{0}".format(s)] = -1.0
                
                else:
                    infos[agent.name]["coop_success_stage_{0}".format(s)] = -1.0
    
            rewards[agent.name] = 0.1 * reward**2

            if second_task_dict is not None:
                if agent.name == "agent_0":
                    done = self.agent_0_playground.done or not self.agent_0_engine.game_on
                    truncated = self.agent_0_playground.done or not self.agent_0_engine.game_on
                else:
                    done = self.agent_1_playground.done or not self.agent_1_engine.game_on
                    truncated = self.agent_1_playground.done or not self.agent_1_engine.game_on
            else:
                done = self.shared_playground.done or not self.shared_engine.game_on
                truncated = self.shared_playground.done or not self.shared_engine.game_on

            dones[agent.name] = done
            truncateds[agent.name] = truncated

        dones["__all__"] = all(dones.values())
        truncateds["__all__"] = all(truncateds.values())
        return rewards, dones, truncateds, infos
    

    def spawn_agents(self, element_coordinates, num_agents, playground, agent_name=None):
        if agent_name is not None:
            assert num_agents == 1, "Only one agent allowed for now"

        sample_pos_agents = random.sample(element_coordinates, num_agents)
        possible_agent_samplers = []
        for i in range(num_agents):
            element_coordinates.remove(sample_pos_agents[i])
            agent_sampler = CoordinateSampler(sample_pos_agents[i], area_shape="rectangle", size=(20, 40))
            possible_agent_samplers.append(agent_sampler)

        
        possible_agent_colors = [(255, 255, 255), (170, 170, 170), (0, 0, 255)]
        agent_ls = []
        for i in range(num_agents):
            if agent_name is not None:
                if agent_name == "agent_0":
                    color = (255, 255, 255)
                elif agent_name == "agent_1":
                    color = (170, 170, 170)
                else:
                    assert False, "Agent name not recognized"
            else:
                color = possible_agent_colors[i]
                    
            agent = BaseAgent(
            controller=External(),
            radius=12,
            mass=15,
            interactive=True, 
            name="agent_{0}".format(i) if agent_name is None else agent_name,
            texture=UniqueCenteredStripeTexture(size=10,
                color=color, color_stripe=(0,0,0), size_stripe=4),
            temporary=True)
            #Makes agents traversable
            categories = 2**3
            for p in agent.parts:
                p.pm_visible_shape.filter = pymunk.ShapeFilter(categories)
            name = "agent_{0}".format(i) if agent_name is None else agent_name
            self.agent_goal_dict[name] = np.zeros(self.num_goals, dtype=int) #Legacy code
            self.agent_ids.add(name)
            agent_ls.append(agent)
      
        for agent, idx in zip(agent_ls, range(self.num_agents)):
            ignore_agents = [agent_ig.parts for agent_ig in agent_ls if agent_ig != agent]
            ignore_agents = [agent_part for agent_ls in ignore_agents for agent_part in agent_ls]
            agent.add_sensor(TopdownSensor(agent.base_platform, fov=360, resolution=self.resolution, max_range=160, normalize=True))
            playground.add_agent(agent, possible_agent_samplers[idx], allow_overlapping=True, max_attempts=10)
    
    def stage_scheduler(self):
        if self.episodes > 0:
            # Calculate rolling averages
            stage1_rolling_avg = np.mean(self.success_rate_dict["stage_1"][-25:])
            stage2_rolling_avg = np.mean(self.success_rate_dict["stage_2"][-25:])
            stage3_rolling_avg = np.mean(self.success_rate_dict["stage_3"][-25:])

            
            # Calculate probabilities
            stage1_probability = max(0.95 - stage1_rolling_avg, 0.025)
            stage2_probability = max(min(stage1_rolling_avg - stage2_rolling_avg, 0.925), 0.025)
            stage3_probability = max(min(stage2_rolling_avg - stage3_rolling_avg, 0.9), 0.025)

        
        else:
            stage1_probability = 0.975
            stage2_probability = 0.025
            stage3_probability = 0.0

        stages_probabilities = [stage1_probability, stage2_probability, stage3_probability]

        #Normalize probabilities
        stages_probabilities = [p / sum(stages_probabilities) for p in stages_probabilities]

        stage = np.random.choice([1, 2, 3], p=stages_probabilities)
        stage = 3 #Hack for now to only sample stage 3 if reward success at all levels
        return stage

    
    def sample_task_tree(self, num_stages, end_conditions, possible_objects, element_coordinates, 
                         env_coordinates, num_agents, num_distractors=0, playground=None, agent_name=None):
        possible_object_types = possible_objects.copy()
        task_dict = {}
        end_condition = random.choice(end_conditions)
        end_condition_object = random.choice(possible_object_types)
        possible_object_types.remove(end_condition_object)
        end_condition_object_shape = end_condition_object[0]
        end_condition_object_color = end_condition_object[1]

        if end_condition == "object_exists":
            end_condition_object = Chest(physical_shape=end_condition_object_shape, radius=10, 
                                        texture=ColorTexture(color=end_condition_object_color, size=10), 
                                        name="end_condition_object", condition_obj=True, temporary=True)
        else:
            end_condition_object = "no_object"
        

        task_dict["end_condition"] = end_condition
        task_dict["end_condition_object"] = end_condition_object

        task_out_objects = [end_condition_object]
        needed_env_objects = []
        assigned_stage_tasks = []
        for s in range(num_stages, 0, -1):
            stage_task_type = self.sample_stage_task(s, num_stages, end_condition, assigned_stage_tasks)
            assigned_stage_tasks.append(stage_task_type)
            needed_in_objects, needed_env_object, condition_obj = self.task_creator(stage_task_type, task_out_objects, possible_object_types, 
                                                                                    s, num_agents, agent_name=agent_name)
            needed_env_objects.append(needed_env_object)
            task_dict["stage_{0}".format(s)] = {}
            task_dict["stage_{0}".format(s)]["task"] = stage_task_type
            task_dict["stage_{0}".format(s)]["in_objects"] = [obj.name for obj in needed_in_objects]
            task_dict["stage_{0}".format(s)]["out_object"] = [obj.name for obj in task_out_objects if obj != "no_object"]
            task_dict["stage_{0}".format(s)]["condition_object"] = condition_obj
            task_out_objects = needed_in_objects
        
        task_dict["num_stages"] = num_stages

        random.shuffle(element_coordinates)
        for object, c in zip(task_out_objects, range(len(task_out_objects))):
            if object != "no_object":
                object_coordinates = CoordinateSampler(element_coordinates[c], area_shape="rectangle", size=(10, 10))

                if playground is None:
                    self.playground.add_element(object, object_coordinates)
                else:
                    playground.add_element(object, object_coordinates)

        random.shuffle(env_coordinates)
        needed_env_objects = [obj for sublist in needed_env_objects for obj in sublist]
        for env_obj, c in zip(needed_env_objects, range(len(needed_env_objects))):
            env_object_coordinates = CoordinateSampler(env_coordinates[c], area_shape="rectangle", size=(10, 10))

            if playground is None:
                self.playground.add_element(env_obj, env_object_coordinates)
            else:
                playground.add_element(env_obj, env_object_coordinates)

        if num_distractors > 0:
            sampled_num_distractors = random.randint(0, num_distractors)
        else:
            sampled_num_distractors = 0
        for d in range(sampled_num_distractors):
            distractor_object = random.choice(possible_object_types)
            possible_object_types.remove(distractor_object)
            distractor_object_shape = distractor_object[0]
            distractor_object_color = distractor_object[1]

            distractor_object = Chest(physical_shape=distractor_object_shape, radius=10, 
                                        texture=ColorTexture(color=distractor_object_color, size=10), 
                                        name="distractor_object_{0}".format(d), condition_obj=False, temporary=True)
            distractor_object_coordinates = CoordinateSampler(element_coordinates[d], area_shape="rectangle", size=(10, 10))

            if playground is None:
                self.playground.add_element(distractor_object, distractor_object_coordinates)
            else:
                playground.add_element(distractor_object, distractor_object_coordinates)

        return task_dict
    
    def task_creator(self, stage_task_type, task_out_objects, possible_object_types, stage, num_agents, agent_name=None):
        needed_in_objects = []
        needed_env_object = []
        if stage_task_type == "crafting":
            for object, n in zip(task_out_objects, range(len(task_out_objects))):
                if n == 0:
                    object_type = random.choice(possible_object_types)
                    possible_object_types.remove(object_type)
                    object_shape = object_type[0]
                    object_color = object_type[1]
                    

                    chest_object = Chest(physical_shape=object_shape, radius=10, 
                                        texture=ColorTexture(color=object_color, size=10),
                                        out_reward=object, 
                                        name="chest_object_{0}".format(stage),
                                        temporary=True)
                    
                    object_type = random.choice(possible_object_types)
                    possible_object_types.remove(object_type)
                    object_shape = object_type[0]
                    object_color = object_type[1]
                    

                    diamond_object = Diamond(chest_object, physical_shape=object_shape, radius=10, 
                                            texture=ColorTexture(color=object_color, size=10), 
                                            name="diamond_object_{0}".format(stage),
                                            temporary=True)
                    
                    needed_in_objects.append(diamond_object)
                    needed_in_objects.append(chest_object)

                    #if task_out_objects[0] == "no_object":
                    #    condition_obj = diamond_object.name
                    #else:
                    #    condition_obj = task_out_objects[0].name

                    condition_obj = chest_object
                else:
                    needed_in_objects.append(object)
                
        elif stage_task_type == "dropoff":
            assert task_out_objects[0] == "no_object"

            dropoff = Chest(physical_shape="rectangle", radius=15,
                            texture=ColorTexture(color=[140, 140, 140], size=15),
                            condition_obj=False, movable=False, graspable=False,
                            dropoff=True,
                            name="dropoff",
                            temporary=True)
            
            object = random.choice(possible_object_types)
            possible_object_types.remove(object)
            object_shape = object[0]
            object_color = object[1]

            dropoff_diamond = Diamond(dropoff, physical_shape=object_shape, radius=10,
                                    texture=ColorTexture(color=object_color, size=10),
                                    name="dropoff_diamond_{0}".format(stage),
                                    temporary=True)
            
            needed_in_objects.append(dropoff_diamond)
            needed_env_object.append(dropoff)
            #condition_obj = dropoff_diamond.name
            condition_obj = dropoff
        
        elif stage_task_type == "activate_landmarks":
            possible_agent_names = []
            if num_agents < 2:
                assert agent_name is not None
                for i in range(2):
                    possible_agent_names.append(agent_name)
            else:
                for a in range(num_agents):
                    possible_agent_names.append("agent_{0}".format(a))
    
            first_agent = random.choice(possible_agent_names)
            possible_agent_names.remove(first_agent)
            second_agent = possible_agent_names[0]


            landmark1 = CustomRewardOnActivation(agent_name=first_agent, radius=15,
                                                physical_shape="rectangle",
                                                texture=ColorTexture(color=[100, 200, 100], size=15),
                                                out_reward=task_out_objects[0],
                                                name="landmark0",
                                                temporary=True)

            landmark2 = CustomRewardOnActivation(agent_name=second_agent, radius=15,
                                                physical_shape="rectangle",
                                                texture=ColorTexture(color=[100, 200, 100], size=15),
                                                out_reward=task_out_objects[1] if len(task_out_objects) > 1 else None,
                                                name="landmark1",
                                                temporary=True)
            
            landmark1.add_partner(landmark2)
            landmark2.add_partner(landmark1)
            
            if len (task_out_objects) > 2:
                needed_in_objects = [obj for obj in task_out_objects[2:]]
            else:
                needed_in_objects = []

            needed_env_object.append(landmark1)
            needed_env_object.append(landmark2)
            condition_obj = landmark1

        elif stage_task_type == "lemon_hunt":
            possible_agent_names = []
            if num_agents < 2:
                assert agent_name is not None
                for i in range(2):
                    possible_agent_names.append(agent_name)
            else:
                for a in range(num_agents):
                    possible_agent_names.append("agent_{0}".format(a))

            lemon_agent = random.choice(possible_agent_names)
            possible_agent_names.remove(lemon_agent)
            agent_name = possible_agent_names[0]

            object = random.choice(possible_object_types)
            possible_object_types.remove(object)
            object_shape = object[0]
            object_color = object[1]

            lemon = Lemon(physical_shape=object_shape, radius=10,
                            texture=object_color,
                            name="lemon_{0}".format(stage), agent_name=lemon_agent,
                            temporary=True)
            
            object = random.choice(possible_object_types)
            possible_object_types.remove(object)
            object_shape = object[0]
            object_color = object[1]

            lemon_dispenser = LemonDispenser(agent_name=agent_name, radius=10,
                                                                texture=ColorTexture(color=object_color, size=15),
                                                                physical_shape=object_shape,
                                                                out_reward=lemon,
                                                                name="lemon_dispenser_{0}".format(stage),
                                                                temporary=True)
            
            needed_in_objects.append(lemon_dispenser)
            condition_obj = lemon
        
        elif stage_task_type == "pressure_plate":
            assert task_out_objects[0] != "no_object"
            
            if num_agents < 2:
                time_limit = 200
            else:
                time_limit = 10

            pressure_plate =  TimedCustomRewardOnActivation(radius=15, time_limit=time_limit, 
                                                            physical_shape="rectangle",
                                                            texture=ColorTexture(color=[25, 100, 145], size=15),
                                                            name="pressure_plate",
                                                            temporary=True)
            
            object = random.choice(possible_object_types)
            possible_object_types.remove(object)
            object_shape = object[0]
            object_color = object[1]

            pressure_plate_in_out = InputOutputMachine(physical_shape="rectangle", radius=15,
                            texture=ColorTexture(color=[20, 40, 170], size=15),
                            condition_obj=True,
                            activation_zone=pressure_plate,
                            name="in_out_machine_pressure_plate", reward=task_out_objects[0],
                            temporary=True)
            
            pressure_plate_diamond = Diamond(pressure_plate_in_out, physical_shape=object_shape, radius=10,
                                    texture=ColorTexture(color=object_color, size=10),
                                    name="pressure_plate_diamond_{0}".format(stage),
                                    temporary=True)
            
            needed_in_objects.append(pressure_plate_diamond)
            for obj in task_out_objects[1:]:
                if obj != "no_object":
                    needed_in_objects.append(obj)  

            needed_env_object.append(pressure_plate)
            needed_env_object.append(pressure_plate_in_out)
            condition_obj = pressure_plate_in_out 
        
        else:
            assert task_out_objects[0] != "no_object"
            in_out_machine = InputOutputMachine(physical_shape="rectangle", radius=15,
                            texture=ColorTexture(color=[50, 50, 200], size=15),
                            condition_obj=True,
                            name="in_out_machine", reward=task_out_objects[0],
                            temporary=True)
            
            object = random.choice(possible_object_types)
            possible_object_types.remove(object)
            object_shape = object[0]
            object_color = object[1]
        
            in_out_machine_diamond = Diamond(in_out_machine, physical_shape=object_shape, radius=10,
                                    texture=ColorTexture(color=object_color, size=10),
                                    name="in_out_machine_diamond_{0}".format(stage),
                                    temporary=True)
            
            needed_in_objects.append(in_out_machine_diamond)
            for obj in task_out_objects[1:]:
                if obj != "no_object":
                    needed_in_objects.append(obj)

            needed_env_object.append(in_out_machine)
            condition_obj = in_out_machine                                            

        return needed_in_objects, needed_env_object, condition_obj

    def sample_stage_task(self, stage, num_stages, end_condition, assigned_stage_tasks):

        if stage == 1 and num_stages > 1:
            if "lemon_hunt" not in assigned_stage_tasks and "double_activate" not in assigned_stage_tasks:
                stage_task = random.choice(["activate_landmarks", "double_activate"])
            elif "lemon_hunt" not in assigned_stage_tasks and "double_activate" in assigned_stage_tasks:
                if "in_out_machine" not in assigned_stage_tasks:
                    stage_task = random.choice(["activate_landmarks", "crafting", "in_out_machine"])
                else:
                    stage_task = random.choice(["activate_landmarks", "crafting"])
            elif "lemon_hunt" in assigned_stage_tasks and "double_activate" not in assigned_stage_tasks:
                if "in_out_machine" not in assigned_stage_tasks:
                    stage_task = random.choice(["activate_landmarks", "double_activate", "in_out_machine", "crafting"])
                else:
                    stage_task = random.choice(["activate_landmarks", "double_activate", "crafting"])
               
           
        elif stage == num_stages and num_stages > 1:
            if end_condition == "no_object":
                stage_task = random.choice(["lemon_hunt", "dropoff", "crafting"])
            else:
                stage_task = random.choice(["crafting", "in_out_machine"])

        elif stage == num_stages and num_stages == 1:
            if end_condition == "no_object":
                stage_task = random.choice(["activate_landmarks", "lemon_hunt"])
            else:
                stage_task = random.choice(["activate_landmarks", "double_activate"])

        else:
            if "crafting" in assigned_stage_tasks:
                if "lemon_hunt" not in assigned_stage_tasks:
                    if "double_activate" not in assigned_stage_tasks:
                        if "in_out_machine" not in assigned_stage_tasks:
                            stage_task = random.choice(["double_activate", "in_out_machine"])
                        else:
                            stage_task = "double_activate"
                    else:
                        if "in_out_machine" not in assigned_stage_tasks:
                            stage_task = random.choice(["crafting", "in_out_machine"])
                        else:
                            stage_task = "crafting"
                else:
                    if "in_out_machine" not in assigned_stage_tasks:
                        stage_task = random.choice(["crafting", "in_out_machine"])
                    else:
                        stage_task = "crafting"
            else:
                if "in_out_machine" not in assigned_stage_tasks:
                    stage_task = random.choice(["crafting", "in_out_machine"])
                else:
                    stage_task = "crafting"
        
        return stage_task
    
    def build_coordinates(self):
        element_coordinates = []
        for w in range(50, self.playground_width, 50):
            x_coord = w
            for h in range(50, self.playground_height, 50):
                y_coord = h
                element_coordinates.append((x_coord, y_coord))
        element_coordinates.remove((self.playground_width // 2, self.playground_height // 2))
        return element_coordinates

    def build_env_coordinates(self):
        env_coordinates = []
        quarter_width = self.playground_width // 4
        quarter_height = self.playground_height // 4

        for i in [1, 3]:
            x_coord = i * quarter_width
            y_coord = i * quarter_height

            env_coordinates.append((x_coord, 20))
            env_coordinates.append((x_coord, int(self.playground_width-20)))

            env_coordinates.append((20, y_coord))
            env_coordinates.append((int(self.playground_height-20), y_coord))
        return env_coordinates
    
    def clip_actions(self, actions, act_idx):
        return np.clip(actions, self.action_space["actuators_action_space"].low[act_idx], 
                       self.action_space["actuators_action_space"].high[act_idx])
    
    def render(self):
        if self.coop:
            image = self.shared_engine.generate_agent_image(self.shared_playground.agents[0], max_size_pg=max(self.playground_height, self.playground_width))
        else:
            image = self.agent_1_engine.generate_agent_image(self.agent_1_playground.agents[0], max_size_pg=max(self.playground_height, self.playground_width))
        return image
       
    def close(self):
        self.engine.terminate()



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    config = {"num_landmarks": 1,
              "num_agents": 2,
              "timelimit": 10000,
              "coop_chance":0.5,
              "message_length": 3,
              "vocab_size": 3,
              "message_penalty": 0.02,
              "seed": 42,
              "stages": 1,
              "forced_coop_rate": 0.0,
              "playground_width": 300,
              "playground_height": 300,
              "single_goal": True,
              "single_reward": False,
              "random_assign": True,
              "new_tasks": True,
              "min_prob": 0.025,
              "max_prob": 0.95,
              "agent_resolution": 128}
    env = CoopCraftingEnv(config)
    print(env.action_space.sample())
    obs = env.reset()
    obs_sampled = env.observation_space.sample()
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
        cv2.waitKey(100)

        for e in range(4):
            #actions = {"agent_0": {"actuators_action_space": torch.Tensor(env.action_space["actuators_action_space"].sample()),
            #                       "message_action_space": torch.Tensor(env.action_space["message_action_space"].sample())},
            #           "agent_1": {"actuators_action_space": torch.Tensor(env.action_space["actuators_action_space"].sample()),
            #                       "message_action_space": torch.Tensor(env.action_space["message_action_space"].sample())}}

            actions = {"agent_0": torch.Tensor(env.action_space.sample()),
                       "agent_1": torch.Tensor(env.action_space.sample())}

            obs, rewards, dones, _, info = env.step(actions)
            print(obs.keys(), rewards.keys(), dones.keys(), info.keys())
            img = env.render()
            cv2.imshow('agent', img)
            cv2.waitKey(100)
