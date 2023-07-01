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
            mass=20,
            name=name,
            **kwargs
        )

        self.chest = chest
        self.graspable = True

class Chest(ActivableByGem):
    def __init__(self, radius, physical_shape, texture, name, condition_obj=True, movable=True,
                  graspable=True, reward=None, dropoff=False, **kwargs):

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

        self.dropoff = dropoff
        self.condition_obj = condition_obj
        if reward != "no_object":
            self.reward = reward
        else:
            self.reward = None

    def _set_pm_collision_type(self):
        for pm_shape in self._pm_shapes:
            pm_shape.collision_type = CollisionTypes.ACTIVABLE_BY_GEM

    def activate(self, activating):
        
        list_remove = None
        elem_add = None

        if activating.elem_activated is self:

            if self.dropoff:
                list_remove = [activating]
            else:
                list_remove = [activating, self]

            if self.condition_obj:
                elem_add = [(self.reward, self.coordinates)]


        return list_remove, elem_add
    
    
class CustomRewardOnActivation(RewardOnActivation):
    def __init__(self, agent_name, out_reward, **kwargs):
        super().__init__(reward=0, **kwargs)
        self.agent_name = agent_name
        self._reward = 0
        self.active = False
        self.spawned = False

        if out_reward != "no_object":
            self.out_reward = out_reward
        else:
            self.out_reward = None

    def activate(self, activating):
        list_remove = None
        elem_add = None

        if activating.name == self.agent_name or activating.name == self.partner_landmark.name:
            if self.spawned is False:
                self.active = True
                self._texture_surface.fill(color=(255, 255, 255))
                if self.partner_landmark.active:
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
    
    def add_partner(self, partner_landmark):
        self.partner_landmark = partner_landmark

    def get_spawn_coordinates(self):
        self_spawn_coordinates_center = (self.coordinates[0][0], self.coordinates[0][1])
        partner_spawn_coordinates_center = (self.partner_landmark.coordinates[0][0], self.partner_landmark.coordinates[0][1])
        self_spawn_coordinates_sampler = CoordinateSampler(self_spawn_coordinates_center, "rectangle", (15, 15))
        partner_spawn_coordinates_sampler = CoordinateSampler(partner_spawn_coordinates_center, "rectangle", (15, 15))
        self_spawn_coordinates = self_spawn_coordinates_sampler.sample()
        partner_spawn_coordinates = partner_spawn_coordinates_sampler.sample()
        return self_spawn_coordinates, partner_spawn_coordinates

    
class LemonDispenser(ActivableElement):
    def __init__(self, out_reward, radius, physical_shape, texture, name, agent_name, **kwargs):

        super().__init__(
            config_key=ElementTypes.CANDY,
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
        self.agent_name = agent_name
        self.out_reward = out_reward
    
    def activate(self, activating):
        list_remove = None
        elem_add = None

        if activating.name == self.agent_name:
            list_remove = [self]
            elem_add = [(self.out_reward, self.coordinates)]

        return list_remove, elem_add
    
    @property
    def terminate_upon_activation(self):
        return False

 

class Lemon(ContactElement):
    def __init__(self, radius, physical_shape, texture, name, agent_name, **kwargs):

        super().__init__(
            config_key=ElementTypes.CANDY,
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
        self.agent_name = agent_name
    
    def activate(self, activating):
        list_remove = None
        elem_add = None

        if activating.name == self.agent_name:
            list_remove = [self]

        return list_remove, elem_add

    @property
    def terminate_upon_activation(self):
        return False
    
class InputOutputMachine(ActivableByGem):
    def __init__(self, radius, physical_shape, texture, name, reward, condition_obj=True, movable=False,
                  graspable=False, **kwargs):

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

    def _set_pm_collision_type(self):
        for pm_shape in self._pm_shapes:
            pm_shape.collision_type = CollisionTypes.ACTIVABLE_BY_GEM

    def activate(self, activating):
        
        list_remove = None
        elem_add = None

        if activating.elem_activated is self:

            if self.condition_obj:
                coordinates = self.get_spawn_coordinates()
                elem_add = [(self.reward, coordinates)]
                list_remove = [activating]


        return list_remove, elem_add

    def get_spawn_coordinates(self):
        self_spawn_coordinates_center = (self.coordinates[0][0], self.coordinates[0][1])
        self_spawn_coordinates_sampler = CoordinateSampler(self_spawn_coordinates_center, "rectangle", (15, 15))
        self_spawn_coordinates = self_spawn_coordinates_sampler.sample()
        return self_spawn_coordinates
 



class CraftingEnv(MultiAgentEnv):
    def __init__(self, config):
        super(CraftingEnv, self).__init__()

        self.config = config
        self.num_goals = config["num_landmarks"]
        self.num_agents = config["num_agents"]
        self.timelimit = config["timelimit"]
        self.playground_height = config["playground_height"]
        self.playground_width = config["playground_width"]
        self.seed = config["seed"]
        self.min_prob = config["min_prob"]
        self.max_prob = config["max_prob"]
        self.episodes = 0
        self.time_steps = 0
        self.truncated = False
        self.playground = GridRooms(size=(self.playground_height, self.playground_width), room_layout=(2, 2), 
                                    random_doorstep_position=False, doorstep_size=80)
        self.agent_ids = set()
        self.single_reward = config["single_reward"]

        if "test" in config.keys():
            self.test = True
        else:
            self.test = False
        
        if "agents_per_env" in config.keys():
            self.num_agents = config["agents_per_env"]

        
        
        self.engine = Engine(
            playground=self.playground, time_limit=(self.timelimit + 1),  
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
        
        self.observation_space = spaces.Box(high=1, low=0, shape=(64, 64, 3), dtype=np.float32)

        self.success_rate_dict = {}
        for s in range(1, 4):
            self.success_rate_dict["stage_{0}".format(s)] = [False]



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
        self.end_condition_object_has_existed = False
        
        #Build element coordinates
        self.build_coordinates()
        self.build_env_coordinates()

        #Possible objects
        possible_objects = [("circle",[255,255,0]),("circle",[0,255,55]),("circle",[255,0,255]),
                        ("rectangle",[255,255,0]),("rectangle",[0,255,255]),("rectangle",[255,0,255]),
                        ("triangle",[255,255,0]),("triangle",[0,255,255]),("triangle",[255,0,255])]
        
        end_conditions = ["no_object", "object_exists"]
        
        #Spawn in agents
        self.spawn_agents()

        self.stage = self.stage_scheduler()
    
        
        #Build the task tree and spawn in the required elements
        self.task_dict = self.sample_task_tree(self.stage, end_conditions, possible_objects)

        info = {}
        self._active_agents = self.playground.agents.copy()

        self.stage_first_reward_dict = {}
        for agent in self._active_agents:
            self.stage_first_reward_dict[agent.name] = {"stage_{0}".format(s): True for s in range(1, self.stage + 1)}
        
        self.engine.elapsed_time = 0
        self.time_steps = 0
        self.episodes += 1

        self.engine.update_observations()
        observations = self.process_obs()
        return observations, info
    
    def compute_reward(self):
        reward = False

        end_condition_type = self.task_dict["end_condition"]

        existing_playground_element_names = [element.name for element in self.playground.elements]


        for s in range(1, self.stage+1):
            if s < self.stage:
                condtion_object = self.task_dict["stage_{0}".format(s)]["condition_object"]
                if condtion_object in existing_playground_element_names:
                    self.success_rate_dict["stage_{0}".format(s)].append(True)
                else:
                    self.success_rate_dict["stage_{0}".format(s)].append(False)
            else:
                if end_condition_type == "object_exists":
                    end_condition_object = self.task_dict["end_condition_object"]
                    if end_condition_object.name in existing_playground_element_names:
                        self.success_rate_dict["stage_{0}".format(s)].append(True)
                        reward = True
                    else:
                        self.success_rate_dict["stage_{0}".format(s)].append(False)
                else:
                    end_condition_object = self.task_dict["stage_{0}".format(s)]["condition_object"]
                    if end_condition_object != "no_object":
                        if end_condition_object in existing_playground_element_names:
                            end_condition_object_exists = True
                            self.end_condition_object_has_exited = True
                        else:
                            end_condition_object_exists = False

                        if end_condition_object_exists is False and self.end_condition_object_has_existed is True:
                            self.success_rate_dict["stage_{0}".format(s)].append(True)
                            reward = True
                        else:
                            self.success_rate_dict["stage_{0}".format(s)].append(False)
                    else:
                        #Shoud only be possible for one stage activate landmarks
                        active_list = []
                        for element in self.playground.elements:
                            if isinstance(element, CustomRewardOnActivation):
                                active_list.append(element.active)
                        if all(active_list):
                            self.success_rate_dict["stage_{0}".format(s)].append(True)
                            reward = True
                        else:
                            self.success_rate_dict["stage_{0}".format(s)].append(False)

        rewards = {}
        dones = {}
        truncateds = {}
        infos = {}
        
        for agent in self._active_agents:
            if reward:
                agent_reward = 1.0
            else:
                agent_reward = 0.0
            rewards[agent.name] = agent_reward

            if self.stage_first_reward_dict[agent.name]["stage_{0}".format(self.stage)] and bool(reward):
                infos[agent.name] = {"success": 1.0, "goal_line": 0.0, "true_goal":  self.agent_goal_dict[agent.name]}
            else:
                infos[agent.name] = {"success": 0.0, "goal_line": 0.0, "true_goal":  self.agent_goal_dict[agent.name]}
            
            for s in range(1, self.stage + 1):
                if self.success_rate_dict["stage_{0}".format(s)][-1]: 
                    if self.stage_first_reward_dict[agent.name]["stage_{0}".format(s)]:
                        infos[agent.name]["success_stage_{0}".format(s)] = 1.0
                        self.stage_first_reward_dict[agent.name]["stage_{0}".format(s)] = False
                    else:
                        infos[agent.name]["success_stage_{0}".format(s)] = 0.0
                else:
                    infos[agent.name]["success_stage_{0}".format(s)] = 0.0
                    
            for s in range(self.stage+1, 4):
                infos[agent.name]["success_stage_{0}".format(s)] = -1.0
    
            if self.single_reward:
                done = bool(reward) or self.playground.done or not self.engine.game_on
            else:
                rewards[agent.name] = 0.1 * reward
                done = self.playground.done or not self.engine.game_on
            
            
            truncated = self.playground.done or not self.engine.game_on
            dones[agent.name] = done
            truncateds[agent.name] = truncated

        dones["__all__"] = all(dones.values())
        truncateds["__all__"] = all(truncateds.values())
        return rewards, dones, truncateds, infos
    
    def spawn_agents(self):
        sample_pos_agents = random.sample(self.element_coordinates, self.num_agents)
        self.element_coordinates.remove(sample_pos_agents[0])
        self.element_coordinates.remove(sample_pos_agents[1])

        agent_sampler_chest = CoordinateSampler(
            sample_pos_agents[0], area_shape="rectangle", size=(20, 40)
        )
        agent_sampler_diamond = CoordinateSampler(
            sample_pos_agents[1], area_shape="rectangle", size=(20, 40)
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
                color=possible_agent_colors[i], color_stripe=(0,0,0), size_stripe=4),
            temporary=True)
            #Makes agents traversable
            categories = 2**3
            for p in agent.parts:
                p.pm_visible_shape.filter = pymunk.ShapeFilter(categories)
            agent_dict["agent_{0}".format(i)] = agent
            self.agent_goal_dict["agent_{0}".format(i)] = np.zeros(self.num_goals, dtype=int)
            self.agent_ids.add("agent_{0}".format(i))
            agent_ls.append(agent)
      
        for agent, idx in zip(agent_ls, range(self.num_agents)):
            ignore_agents = [agent_ig.parts for agent_ig in agent_ls if agent_ig != agent]
            ignore_agents = [agent_part for agent_ls in ignore_agents for agent_part in agent_ls]
            agent.add_sensor(TopdownSensor(agent.base_platform, fov=360, resolution=64, normalize=True))
            self.playground.add_agent(agent, possible_agent_samplers[idx], allow_overlapping=True, max_attempts=10)
    
    def stage_scheduler(self):
        if self.episodes > 1:
            # Calculate rolling averages
            stage1_rolling_avg = np.mean(self.success_rate_dict["stage_1"][-25:])
            stage2_rolling_avg = np.mean(self.success_rate_dict["stage_2"][-25:])
            stage3_rolling_avg = np.mean(self.success_rate_dict["stage_3"][-25:])

            
            # Calculate probabilities
            stage1_probability = max(0.95 - stage1_rolling_avg, 0.025)
            stage2_probability = max(min(stage1_rolling_avg - stage2_rolling_avg, 0.925), 0.025)
            stage3_probability = max(min(stage2_rolling_avg - stage3_rolling_avg, 0.9), 0.025)

        
        else:
            stage1_probability = 0.95
            stage2_probability = 0.025
            stage3_probability = 0.025

        stages_probabilities = [stage1_probability, stage2_probability, stage3_probability]

        #Normalize probabilities
        stages_probabilities = [p / sum(stages_probabilities) for p in stages_probabilities]

        stage = np.random.choice([1, 2, 3], p=stages_probabilities)
        return stage

    
    def sample_task_tree(self, num_stages, end_conditions, possible_objects):
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
            needed_in_objects, needed_env_object, condition_obj = self.task_creator(stage_task_type, task_out_objects, possible_object_types, s)
            needed_env_objects.append(needed_env_object)
            task_dict["stage_{0}".format(s)] = {}
            task_dict["stage_{0}".format(s)]["task"] = stage_task_type
            task_dict["stage_{0}".format(s)]["in_objects"] = [obj.name for obj in needed_in_objects]
            task_dict["stage_{0}".format(s)]["out_object"] = [obj.name for obj in task_out_objects if obj != "no_object"]
            task_dict["stage_{0}".format(s)]["condition_object"] = condition_obj
            task_out_objects = needed_in_objects
        
        task_dict["num_stages"] = num_stages

        random.shuffle(self.element_coordinates)
        for object, c in zip(task_out_objects, range(len(task_out_objects))):
            if object != "no_object":
                object_coordinates = CoordinateSampler(self.element_coordinates[c], area_shape="rectangle", size=(10, 10))
                self.playground.add_element(object, object_coordinates)

        random.shuffle(self.env_coordinates)
        needed_env_objects = [obj for sublist in needed_env_objects for obj in sublist]
        for env_obj, c in zip(needed_env_objects, range(len(needed_env_objects))):
            env_object_coordinates = CoordinateSampler(self.env_coordinates[c], area_shape="rectangle", size=(10, 10))
            self.playground.add_element(env_obj, env_object_coordinates)

        return task_dict
    
    def task_creator(self, stage_task_type, task_out_objects, possible_object_types, stage):
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
                                        reward=object, 
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
                    condition_obj = diamond_object.name
                else:
                    needed_in_objects.append(object)
                
        elif stage_task_type == "dropoff":
            assert task_out_objects[0] == "no_object"

            dropoff = Chest(physical_shape="rectangle", radius=15,
                            texture=MultipleCenteredStripesTexture(color_1=[200, 200, 200], color_2=[20, 20, 20], size=15, n_stripes=3),
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
            condition_obj = dropoff_diamond.name
        
        elif stage_task_type == "activate_landmarks":
            possible_agent_names = ["agent_0", "agent_1"] #TODO: Make more general
            first_agent = random.choice(possible_agent_names)
            possible_agent_names.remove(first_agent)
            second_agent = possible_agent_names[0]


            landmark1 = CustomRewardOnActivation(agent_name=first_agent, radius=15,
                                                texture=MultipleCenteredStripesTexture(color_1=[255, 0, 0], color_2=[255, 255, 255], size=15, n_stripes=3),
                                                out_reward=task_out_objects[0],
                                                name="landmark0",
                                                temporary=True)

            landmark2 = CustomRewardOnActivation(agent_name=second_agent, radius=15,
                                                texture=MultipleCenteredStripesTexture(color_1=[0, 255, 0], color_2=[255, 255, 255], size=15, n_stripes=3),
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
            if task_out_objects[0] != [] and task_out_objects[0] != "no_object":
                condition_obj = task_out_objects[0].name
            else:
                condition_obj = "no_object"

        elif stage_task_type == "lemon_hunt":
            possible_agent_names = ["agent_0", "agent_1"] #TODO: Make more general
            lemon_agent = random.choice(possible_agent_names)
            possible_agent_names.remove(lemon_agent)
            agent_name = possible_agent_names[0]

            object = random.choice(possible_object_types)
            possible_object_types.remove(object)
            object_shape = object[0]
            object_color = object[1]

            lemon = Lemon(physical_shape=object_shape, radius=10,
                            texture=ColorTexture(color=object_color, size=10),
                            name="lemon_{0}".format(stage), agent_name=lemon_agent,
                            temporary=True)
            
            object = random.choice(possible_object_types)
            possible_object_types.remove(object)
            object_shape = object[0]
            object_color = object[1]

            lemon_dispenser = LemonDispenser(agent_name=agent_name, radius=10,
                                                                texture=ColorTexture(color=object_color, size=10),
                                                                physical_shape=object_shape,
                                                                out_reward=lemon,
                                                                name="lemon_dispenser_{0}".format(stage),
                                                                temporary=True)
            needed_in_objects.append(lemon_dispenser)
            condition_obj = lemon.name
        
        else:
            assert task_out_objects[0] != "no_object"
            in_out_machine = InputOutputMachine(physical_shape="rectangle", radius=15,
                            texture=MultipleCenteredStripesTexture(color_1=[100, 100, 100], color_2=[255, 255, 255], size=15, n_stripes=3),
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
            needed_env_object.append(in_out_machine)
            condition_obj = task_out_objects[0].name                                            

        return needed_in_objects, needed_env_object, condition_obj

    def sample_stage_task(self, stage, num_stages, end_condition, assigned_stage_tasks):
        if stage == 1 and num_stages > 1:
            stage_task = "activate_landmarks"
        elif stage == num_stages and num_stages > 1:
            if end_condition == "no_object":
                stage_task = random.choice(["lemon_hunt", "dropoff", "crafting"])
            else:
                stage_task = random.choice(["crafting", "in_out_machine"])
        elif stage == num_stages and num_stages == 1:
            if end_condition == "no_object":
                stage_task = random.choice(["activate_landmarks", "lemon_hunt"])
            else:
                stage_task = "activate_landmarks"
        else:
            if "in_out_machine" not in assigned_stage_tasks:
                stage_task = random.choice(["crafting", "in_out_machine"])
            else:
                stage_task = random.choice(["crafting"])
        
        return stage_task
    
    def build_coordinates(self):
        self.element_coordinates = []
        for w in range(50, self.playground_width, 50):
            x_coord = w
            for h in range(50, self.playground_height, 50):
                y_coord = h
                self.element_coordinates.append((x_coord, y_coord))
        self.element_coordinates.remove((self.playground_width // 2, self.playground_height // 2))

    def build_env_coordinates(self):
        self.env_coordinates = []
        quarter_width = self.playground_width // 4
        quarter_height = self.playground_height // 4

        for i in [1, 3]:
            x_coord = i * quarter_width
            y_coord = i * quarter_height

            self.env_coordinates.append((x_coord, 20))
            self.env_coordinates.append((x_coord, int(self.playground_width-20)))

            self.env_coordinates.append((20, y_coord))
            self.env_coordinates.append((int(self.playground_height-20), y_coord))
    
    def clip_actions(self, actions, act_idx):
        return np.clip(actions, self.action_space.low[act_idx], self.action_space.high[act_idx])
    
    def render(self):
         image = self.engine.generate_agent_image(self.playground.agents[0], max_size_pg=max(self.playground_height, self.playground_width))
         return image
       
    def close(self):
        self.engine.terminate()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    config = {"num_landmarks": 1,
              "num_agents": 2,
              "timelimit": 10000,
              "coop_chance":0.0,
              "message_length": 3,
              "vocab_size": 3,
              "message_penalty": 0.02,
              "seed": 42,
              "playground_width": 300,
              "playground_height": 300,
              "single_goal": True,
              "single_reward": False,
              "random_assign": True,
              "min_prob": 0.025,
              "max_prob": 0.95,}
    env = CraftingEnv(config)
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
        cv2.waitKey(10)

        for e in range(4):
            #actions = {"agent_0": {"actuators_action_space": torch.Tensor(env.action_space["actuators_action_space"].sample()),
            #                       "message_action_space": torch.Tensor(env.action_space["message_action_space"].sample())},
            #           "agent_1": {"actuators_action_space": torch.Tensor(env.action_space["actuators_action_space"].sample()),
            #                       "message_action_space": torch.Tensor(env.action_space["message_action_space"].sample())}}

            actions = {"agent_0": torch.Tensor(env.action_space.sample()),
                       "agent_1": torch.Tensor(env.action_space.sample())}

            obs, rewards, dones, _, info = env.step(actions)
            img = env.render()
            cv2.imshow('agent', img)
            cv2.waitKey(10)

