from crafting_env import CraftingEnv, CustomRewardOnActivation

from simple_playgrounds.playgrounds.layouts import SingleRoom
from simple_playgrounds.engine import Engine
from simple_playgrounds.agents.parts.controllers import Keyboard
from simple_playgrounds.agents.agents import BaseAgent, HeadAgent
from simple_playgrounds.agents.sensors.topdown_sensors import TopdownSensor

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def plt_image(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()




def compute_reward(playground, task_dict, stage, agent_goal_dict, success_rate_dict, 
                   stage_first_reward_dict, end_condition_object_has_existed):

        end_condition_type = task_dict["end_condition"]

        existing_playground_element_names = [element.name for element in playground.elements]


        for s in range(1, stage+1):
            if s < stage:
                condition_object = task_dict["stage_{0}".format(s)]["condition_object"]
                if condition_object in existing_playground_element_names:
                    success_rate_dict["stage_{0}".format(s)].append(True)
                else:
                    success_rate_dict["stage_{0}".format(s)].append(False)
            else:
                if end_condition_type == "object_exists":
                    end_condition_object = task_dict["end_condition_object"]
                    if end_condition_object.name in existing_playground_element_names:
                        success_rate_dict["stage_{0}".format(s)].append(True)
                    else:
                        success_rate_dict["stage_{0}".format(s)].append(False)
                else:
                    condition_object = task_dict["stage_{0}".format(s)]["condition_object"]
                    if condition_object != "no_object":
                        if condition_object in existing_playground_element_names:
                            end_condition_object_exists = True
                            end_condition_object_has_existed = True
                        else:
                            end_condition_object_exists = False
                    
                        if end_condition_object_exists is False and end_condition_object_has_existed is True:
                            success_rate_dict["stage_{0}".format(s)].append(True)
                        else:
                            success_rate_dict["stage_{0}".format(s)].append(False)
                    else:
                        #Shoud only be possible for one stage activate landmarks
                        active_list = []
                        for element in playground.elements:
                            if isinstance(element, CustomRewardOnActivation):
                                active_list.append(element.active)
                        if all(active_list):
                            success_rate_dict["stage_{0}".format(s)].append(True)
                        else:
                            success_rate_dict["stage_{0}".format(s)].append(False)
        
        stage_success = []
        for stage in range(1, stage + 1):
            stage_success.append(any(success_rate_dict["stage_{0}".format(stage)]))
        
        reward = sum(stage_success)
        

        rewards = {}
        infos = {}
        
        for agent in playground.agents:
            rewards[agent.name] = reward

            if stage_first_reward_dict[agent.name]["stage_{0}".format(stage)] and int(reward) == stage:
                print("Reward")
                infos[agent.name] = {"success": 1.0, "goal_line": 0.0, "true_goal":  agent_goal_dict[agent.name]}
            else:
                infos[agent.name] = {"success": 0.0, "goal_line": 0.0, "true_goal":  agent_goal_dict[agent.name]}
            
            for s in range(1, stage + 1):
                if success_rate_dict["stage_{0}".format(s)][-1]: 
                    if stage_first_reward_dict[agent.name]["stage_{0}".format(s)]:
                        infos[agent.name]["success_stage_{0}".format(s)] = 1.0
                        stage_first_reward_dict[agent.name]["stage_{0}".format(s)] = False
                    else:
                        infos[agent.name]["success_stage_{0}".format(s)] = 0.0
                else:
                    infos[agent.name]["success_stage_{0}".format(s)] = 0.0
                    
            for s in range(stage+1, 4):
                infos[agent.name]["success_stage_{0}".format(s)] = -1.0
    
            rewards[agent.name] = 0.1 * reward

        return rewards, infos, end_condition_object_has_existed


if __name__ == '__main__':
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
    base_env = CraftingEnv(config)

    playground = SingleRoom(size=(config["playground_width"], config["playground_height"]), wall_type='light')
    engine = Engine(time_limit=config["timelimit"], playground=playground, screen=True)

    agent = BaseAgent(controller=Keyboard(), interactive=True, name="agent_0")
    agent.add_sensor(TopdownSensor(agent.base_platform, max_range=150, resolution=128))
    playground.add_agent(agent)

    element_coordinates = [(20, 20), (20, 280), (280, 20), (280, 280), (60, 60), (60, 240), (240, 60), (240, 240),
                           (100, 100), (100, 200), (200, 100), (200, 200), (150, 150)]
    env_coordinates = [(150, 100), (100, 100), (200, 100), (250, 100), (50, 100)]

    possible_objects = [("circle",[255,255,0]),("circle",[0,255,55]),("circle",[255,0,255]),
                        ("rectangle",[255,255,0]),("rectangle",[0,255,255]),("rectangle",[255,0,255]),
                        ("triangle",[255,255,0]),("triangle",[0,255,255]),("triangle",[255,0,255])]
        
    end_conditions = ["no_object", "object_exists"]

    task_dict = base_env.sample_task_tree(3, end_conditions, possible_objects, element_coordinates, env_coordinates, 3, playground)

    success_rate_dict = {}
    for s in range(1, 4):
        success_rate_dict["stage_{0}".format(s)] = [False]
    
    end_condition_object_has_existed = False
    stage_first_reward_dict = {}
    stage_first_reward_dict[agent.name] = {"stage_{0}".format(s): True for s in range(1, 4)}

    agent_goal_dict = {}
    agent_goal_dict[agent.name] = np.zeros(config["num_landmarks"])
    print(task_dict)
    time.sleep(10)


    while engine.game_on:

        engine.update_screen()

        actions = {}
        for agent in engine.agents:
            actions[agent] = agent.controller.generate_actions()

        terminate = engine.step(actions)
        engine.update_observations()
        rewards, infos, end_condition_object_has_existed = compute_reward(playground, task_dict, 3, agent_goal_dict, success_rate_dict, 
                                        stage_first_reward_dict, end_condition_object_has_existed)
        print(rewards, infos)

        cv2.imshow('agent', engine.generate_agent_image(agent, max_size_pg=300))
        cv2.waitKey(20)
        
        if terminate:
            print(task_dict)
            engine.terminate()

