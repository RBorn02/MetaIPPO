import torch
import yaml
import imageio
import csv

import numpy as np
from collections.abc import Mapping

COMM_ENVS = ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm", "TreasureHuntComm", "CoopCraftingEnvComm"]

def build_storage(config, env):
    """Builds a dict for the storage of the rollout data for each policy"""
    num_steps = config["rollout_steps"] // (config["num_workers"] * config["env_config"]["num_envs"])
    num_envs = config["env_config"]["num_envs"]
    device = config["device"]
    num_layers = config["model_config"]["lstm_layers"]
    hidden_size = config["model_config"]["lstm_hidden_size"]
    message_length = config["env_config"]["message_length"]

    storage = {}
    for a in range(config["env_config"]["num_agents"]):
        storage["agent_{0}".format(a)] = {
                    "obs": torch.zeros((num_steps, num_envs) + env.observation_space.shape).to(device),
                    "dones":torch.zeros((num_steps, num_envs)).to(device),
                    "next_lstm_state":(
                                        torch.zeros(num_layers, num_envs, hidden_size).to(device),
                                        torch.zeros(num_layers, num_envs, hidden_size).to(device),
                                        ),
                    "rewards":torch.zeros((num_steps, num_envs)).to(device),
                    "last_rewards":torch.zeros((num_steps, num_envs)).to(device),
                    "actions":torch.zeros((num_steps, num_envs) + env.action_space_shape).to(device),
                    "message_in":torch.zeros((num_steps, num_envs, message_length)).to(device),
                    "last_actions":torch.zeros((num_steps, num_envs) + env.action_space_shape).to(device),
                    "values":torch.zeros((num_steps, num_envs)).to(device),
                    "logprobs":torch.zeros((num_steps, num_envs)).to(device),
                    "contact":torch.zeros((num_steps, num_envs)).to(device),
                    "time_till_end":torch.zeros((num_steps, num_envs)).to(device),
                    }
              
    return storage

def build_storage_pop(config, env, agent_ids):
    """Builds a dict for the storage of the rollout data for each policy"""
    num_steps = config["rollout_steps"] // (config["num_workers"] * config["env_config"]["num_envs"])
    num_envs = config["env_config"]["num_envs"]
    device = config["device"]
    num_layers = config["model_config"]["lstm_layers"]
    hidden_size = config["model_config"]["lstm_hidden_size"]
    message_length = config["env_config"]["message_length"]


    storage = {}
    for a in agent_ids:
        storage["agent_{0}".format(a)] = {
                    "obs": torch.zeros((num_steps, num_envs) + env.observation_space.shape).to(device),
                    "dones":torch.zeros((num_steps, num_envs)).to(device),
                    #"is_training":torch.zeros((1, num_envs)).to(device),
                    #"collected_env_steps":torch.zeros((num_envs)).to(device),
                    "next_lstm_state":(
                                        torch.zeros(num_layers, num_envs, hidden_size).to(device),
                                        torch.zeros(num_layers, num_envs, hidden_size).to(device),
                                        ),
                    "rewards":torch.zeros((num_steps, num_envs)).to(device),
                    "last_rewards":torch.zeros((num_steps, num_envs)).to(device),
                    "actions":torch.zeros((num_steps, num_envs) + env.action_space_shape).to(device),
                    "message_in":torch.zeros((num_steps, num_envs, message_length)).to(device),
                    "last_actions":torch.zeros((num_steps, num_envs) + env.action_space_shape).to(device),
                    "values":torch.zeros((num_steps, num_envs)).to(device),
                    "logprobs":torch.zeros((num_steps, num_envs)).to(device),
                    "contact":torch.zeros((num_steps, num_envs)).to(device),
                    "time_till_end":torch.zeros((num_steps, num_envs)).to(device),
                    }
              
    return storage

def get_init_tensors(config, storage, env, agent):
    num_steps = config["rollout_steps"]
    num_envs = config["env_config"]["num_envs"]
    device = config["device"]

    action = torch.zeros((1, num_envs) + env.action_space.shape).to(device)
    logprob = torch.zeros((1, num_envs)).to(device)
    value = torch.zeros((1, num_envs)).to(device)
    next_lstm_state = (torch.zeros_like(storage[agent]["next_lstm_state"][0]), torch.zeros_like(storage[agent]["next_lstm_state"][1]))
    return action, logprob, value, next_lstm_state


def truncate_storage(storage, config):
    """Get the actual training data for each environment and truncate to steps_per_env"""
    rollout_steps = config["rollout_steps"] // config["num_workers"]
    next_step_storage = {"agent_{0}".format(a):{} for a in range(config["env_config"]["num_agents"])}
    num_envs = config["env_config"]["num_envs"]
    steps_per_env = int(rollout_steps // num_envs)
    for agent in storage.keys():
        for key in storage[agent].keys():
            if key in ["obs", "dones", "rewards", "actions", "values", "logprobs"]:
                shape = storage[agent][key].shape
                next_step_storage[agent][key] = storage[agent][key][storage[agent]["is_training"].bool()][rollout_steps:rollout_steps + num_envs].reshape((-1,) + shape[1:])
                storage[agent][key] = storage[agent][key][storage[agent]["is_training"].bool()].reshape((-1,) + shape[1:])
            else:
                continue
    return storage, next_step_storage

def build_storage_from_batch(batch, config):
    """Takes the batch returned by multiprocessing queue and builds a storage dict"""
    storage_out = {}
    next_obs = {}
    next_dones = {}
    next_messages = {}
    success_rate = {}
    achieved_goal = {}
    achieved_goal_success = {}
    next_contact = {}
    next_time_till_end = {}
    stage_success_info = {}
    for a in range(config["env_config"]["num_agents"]):
        agent = "agent_{0}".format(a)
        storage_out[agent] = {}
        storage_out[agent]["obs"] = torch.cat([batch[i][0][agent]["obs"] for i in range(len(batch))], dim=1)
        storage_out[agent]["dones"] = torch.cat([batch[i][0][agent]["dones"] for i in range(len(batch))], dim=1)
        storage_out[agent]["next_lstm_state"] = (torch.cat([batch[i][0][agent]["next_lstm_state"][0] for i in range(len(batch))], dim=1),
                                                 torch.cat([batch[i][0][agent]["next_lstm_state"][1] for i in range(len(batch))], dim=1))
        storage_out[agent]["initial_lstm_state"] = (torch.cat([batch[i][0][agent]["initial_lstm_state"][0] for i in range(len(batch))], dim=1),
                                                    torch.cat([batch[i][0][agent]["initial_lstm_state"][1] for i in range(len(batch))], dim=1))
        storage_out[agent]["rewards"] = torch.cat([batch[i][0][agent]["rewards"] for i in range(len(batch))], dim=1)
        storage_out[agent]["last_rewards"] = torch.cat([batch[i][0][agent]["last_rewards"] for i in range(len(batch))], dim=1)
        storage_out[agent]["actions"] = torch.cat([batch[i][0][agent]["actions"] for i in range(len(batch))], dim=1)
        storage_out[agent]["last_actions"] = torch.cat([batch[i][0][agent]["last_actions"] for i in range(len(batch))], dim=1)
        storage_out[agent]["message_in"] = torch.cat([batch[i][0][agent]["message_in"] for i in range(len(batch))], dim=1)
        storage_out[agent]["values"] = torch.cat([batch[i][0][agent]["values"] for i in range(len(batch))], dim=1)
        storage_out[agent]["logprobs"] = torch.cat([batch[i][0][agent]["logprobs"] for i in range(len(batch))], dim=1)
        storage_out[agent]["contact"] = torch.cat([batch[i][0][agent]["contact"] for i in range(len(batch))], dim=1)
        storage_out[agent]["time_till_end"] = torch.cat([batch[i][0][agent]["time_till_end"] for i in range(len(batch))], dim=1)

        next_obs[agent] = torch.cat([batch[i][1][agent] for i in range(len(batch))], dim=1)
        next_dones[agent] = torch.cat([batch[i][2][agent] for i in range(len(batch))], dim=1)
        success_rate[agent] = sum([batch[i][3][agent] for i in range(len(batch))])
        achieved_goal[agent] = torch.sum(torch.cat([batch[i][4][agent].unsqueeze(dim=0) for i in range(len(batch))], dim=0), dim=0)
        achieved_goal_success[agent] = torch.sum(torch.cat([batch[i][5][agent].unsqueeze(dim=0) for i in range(len(batch))], dim=0), dim=0)
        next_contact[agent] = torch.cat([batch[i][6][agent] for i in range(len(batch))], dim=1)
        next_time_till_end[agent] = torch.cat([batch[i][7][agent] for i in range(len(batch))], dim=1)

        if config["env_config"]["env_name"] in ["CraftingEnv", "CoopCraftingEnvComm", "CoopCraftingEnv", "TestCraftingEnv"]:
            stage_success_info[agent] = {}
            for s in range(1, config["env_config"]["stages"] + 1):
                num_stage_sampled = sum([batch[i][8][agent]["stage_{0}".format(s)]["average_success"][0] for i in range(len(batch))])
                num_stage_success = sum([batch[i][8][agent]["stage_{0}".format(s)]["average_success"][1] for i in range(len(batch))])
                

                if config["env_config"]["env_name"] in ["CoopCraftingEnv", "CoopCraftingEnvComm", "TestCraftingEnv"]:
                    num_coop_stage_sampled = sum([batch[i][8][agent]["stage_{0}".format(s)]["coop_success"][0] for i in range(len(batch))])
                    num_coop_stage_success = sum([batch[i][8][agent]["stage_{0}".format(s)]["coop_success"][1] for i in range(len(batch))])

                    num_single_stage_sampled = sum([batch[i][8][agent]["stage_{0}".format(s)]["single_success"][0] for i in range(len(batch))])
                    num_single_stage_success = sum([batch[i][8][agent]["stage_{0}".format(s)]["single_success"][1] for i in range(len(batch))])

                    stage_success_info[agent]["stage_{0}".format(s)] = {"average_success": (num_stage_sampled, num_stage_success),
                                                                        "coop_success": (num_coop_stage_sampled, num_coop_stage_success),
                                                                        "single_success": (num_single_stage_sampled, num_single_stage_success)}
                else:
                    stage_success_info[agent]["stage_{0}".format(s)] = {"average_success": (num_stage_sampled, num_stage_success)}



    
        if config["env_config"]["env_name"] in COMM_ENVS:
            next_messages[agent] = torch.cat([batch[i][9][agent] for i in range(len(batch))], dim=1)
    
    if config["env_config"]["env_name"] in COMM_ENVS:
        return storage_out, next_obs, next_messages, next_dones, success_rate, achieved_goal, achieved_goal_success, next_contact, next_time_till_end, stage_success_info
    else:
        return storage_out, next_obs, next_dones, success_rate, achieved_goal, achieved_goal_success, next_contact, next_time_till_end, stage_success_info
    

def build_storage_from_batch_pop(batch, config):
    """Takes the batch returned by multiprocessing queue and builds a storage dict"""
    storage_out = {}
    next_obs = {}
    next_dones = {}
    next_messages = {}
    success_rate = {}
    achieved_goal = {}
    achieved_goal_success = {}
    next_contact = {}
    stage_success_info = {}
    for a in range(config["env_config"]["num_agents"]):
        agent = "agent_{0}".format(a)
        storage_out[agent] = {}
        agent_batch = []
        for i in range(len(batch)):
            if agent in batch[i][0].keys():
                agent_batch.append(batch[i])
        storage_out[agent]["obs"] = torch.cat([agent_batch[i][0][agent]["obs"] for i in range(len(agent_batch))], dim=1)
        storage_out[agent]["dones"] = torch.cat([agent_batch[i][0][agent]["dones"] for i in range(len(agent_batch))], dim=1)
        storage_out[agent]["next_lstm_state"] = (torch.cat([agent_batch[i][0][agent]["next_lstm_state"][0] for i in range(len(agent_batch))], dim=1),
                                                    torch.cat([agent_batch[i][0][agent]["next_lstm_state"][1] for i in range(len(agent_batch))], dim=1))
        storage_out[agent]["initial_lstm_state"] = (torch.cat([agent_batch[i][0][agent]["initial_lstm_state"][0] for i in range(len(agent_batch))], dim=1),
                                                    torch.cat([agent_batch[i][0][agent]["initial_lstm_state"][1] for i in range(len(agent_batch))], dim=1))
        storage_out[agent]["rewards"] = torch.cat([agent_batch[i][0][agent]["rewards"] for i in range(len(agent_batch))], dim=1)
        storage_out[agent]["last_rewards"] = torch.cat([agent_batch[i][0][agent]["last_rewards"] for i in range(len(agent_batch))], dim=1)
        storage_out[agent]["actions"] = torch.cat([agent_batch[i][0][agent]["actions"] for i in range(len(agent_batch))], dim=1)
        storage_out[agent]["last_actions"] = torch.cat([agent_batch[i][0][agent]["last_actions"] for i in range(len(agent_batch))], dim=1)
        storage_out[agent]["message_in"] = torch.cat([agent_batch[i][0][agent]["message_in"] for i in range(len(agent_batch))], dim=1)
        storage_out[agent]["values"] = torch.cat([agent_batch[i][0][agent]["values"] for i in range(len(agent_batch))], dim=1)
        storage_out[agent]["logprobs"] = torch.cat([agent_batch[i][0][agent]["logprobs"] for i in range(len(agent_batch))], dim=1)
        storage_out[agent]["contact"] = torch.cat([agent_batch[i][0][agent]["contact"] for i in range(len(agent_batch))], dim=1)

        next_obs[agent] = torch.cat([agent_batch[i][1][agent] for i in range(len(agent_batch))], dim=1)
        next_dones[agent] = torch.cat([agent_batch[i][2][agent] for i in range(len(agent_batch))], dim=1)
        success_rate[agent] = sum([agent_batch[i][3][agent] for i in range(len(agent_batch))])
        achieved_goal[agent] = torch.sum(torch.cat([agent_batch[i][4][agent].unsqueeze(dim=0) for i in range(len(agent_batch))], dim=0), dim=0)
        achieved_goal_success[agent] = torch.sum(torch.cat([agent_batch[i][5][agent].unsqueeze(dim=0) for i in range(len(agent_batch))], dim=0), dim=0)
        next_contact[agent] = torch.cat([agent_batch[i][6][agent] for i in range(len(agent_batch))], dim=1)

        if config["env_config"]["env_name"] in ["CraftingEnv", "CraftingEnvComm", "CoopCraftingEnv"]:
            stage_success_info[agent] = {}
            for s in range(1, config["env_config"]["stages"]+1):
                num_stage_sampled = sum([agent_batch[i][7][agent]["stage_{0}".format(s)][0] for i in range(len(agent_batch))])
                num_stage_success = sum([agent_batch[i][7][agent]["stage_{0}".format(s)][1] for i in range(len(agent_batch))])
                stage_success_info[agent]["stage_{0}".format(s)] = (num_stage_sampled, num_stage_success)

        if config["env_config"]["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm", "TreasureHuntComm"]:
            next_messages[agent] = torch.cat([agent_batch[i][7][agent] for i in range(len(agent_batch))], dim=1)
        
    if config["env_config"]["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm", "TreasureHuntComm"]:
        return storage_out, next_obs, next_messages, next_dones, success_rate, achieved_goal, achieved_goal_success, next_contact, stage_success_info
    else:
        return storage_out, next_obs, next_dones, success_rate, achieved_goal, achieved_goal_success, next_contact, stage_success_info
    
    



def move_tensors_to_cpu(inputs_list):
    for input, idx in zip(inputs_list, range(len(inputs_list))):
        if isinstance(input, torch.Tensor):
            input.cpu()
        elif isinstance(input, Mapping):
            for key, value in input.items():
                if isinstance(value, torch.Tensor):
                    input[key] = value.cpu()
                elif isinstance(value, Mapping):
                    for key2, value2 in value.items():
                        if isinstance(value2, torch.Tensor):
                            input[key][key2] = value2.cpu()
                        else:
                            obj_1 = value2[0]
                            obj_2 = value2[1]
                            if isinstance(obj_1, torch.Tensor):
                                input[key][key2] = (obj_1.cpu(), obj_2.cpu())

def move_tensors_to_gpu(inputs_list):
    for input, idx in zip(inputs_list, range(len(inputs_list))):
        if isinstance(input, torch.Tensor):
            input.cuda()
        elif isinstance(input, Mapping):
            for key, value in input.items():
                if isinstance(value, torch.Tensor):
                    input[key] = value.cuda()
                elif isinstance(value, Mapping):
                    for key2, value2 in value.items():
                        if isinstance(value2, torch.Tensor):
                            input[key][key2] = value2.cuda()
                        else:
                            obj_1 = value2[0]
                            obj_2 = value2[1]
                            if isinstance(obj_1, torch.Tensor):
                                input[key][key2] = (obj_1.cuda(), obj_2.cuda())
        




            


def build_config(args):
    """Builds a config dict from the args, or from a yaml file if config is given"""
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {}
        config["env_config"] = {}
        config["env_config"]["env_name"] = args.env_name
        config["env_config"]["num_envs"] = args.num_envs
        config["env_config"]["num_agents"] = args.num_agents
        config["env_config"]["num_landmarks"] = args.num_landmarks
        config["env_config"]["timelimit"] = args.time_limit
        config["env_config"]["coop_chance"] = args.coop_chance
        config["env_config"]["seed"] = args.seed
        config["env_config"]["single_goal"] = args.single_goal
        config["env_config"]["single_reward"] = args.single_reward
        config["env_config"]["vocab_size"] = args.vocab_size
        config["env_config"]["message_length"] = args.message_length
        config["env_config"]["random_assign"] = args.random_assign
        config["env_config"]["min_prob"] = args.min_prob
        config["env_config"]["max_prob"] = args.max_prob
        config["env_config"]["playground_height"] = args.playground_height
        config["env_config"]["playground_width"] = args.playground_width
        config["env_config"]["agent_resolution"] = args.agent_resolution
        config["env_config"]["stages"] = args.stages
        config["env_config"]["new_tasks"] = args.new_tasks
        config["env_config"]["forced_coop_rate"] = args.forced_coop_rate
        
        config["pretrained"] = args.pretrained
        config["total_steps"] = args.total_steps
        config["rollout_steps"] = args.rollout_steps
        config["batch_size"] = args.batch_size
        config["seed"] = args.seed
        config["anneal_lr"] = args.anneal_lr
        config["lr"] = args.lr
        config["gae"] = args.gae
        config["gae_lambda"] = args.gae_lambda
        config["gamma"] = args.gamma
        config["num_minibatches"] = args.num_minibatches
        config["update_epochs"] = args.update_epochs
        config["norm_adv"] = args.norm_adv
        config["clip_coef"] = args.clip_coef
        config["clip_vloss"] = args.clip_vloss
        config["ent_coef"] = args.ent_coef
        config["vf_coef"] = args.vf_coef
        config["max_grad_norm"] = args.max_grad_norm
        config["target_kl"] = args.target_kl
        config["device"] = args.device
        config["num_workers"] = args.num_workers
        config["record_video_every"] = args.record_video_every
        config["debug"] = args.debug
        config["run_name"] = args.run_name

        config["model_config"] = {}
        config["model_config"]["lstm_layers"] = args.lstm_layers
        config["model_config"]["lstm_in_size"] = args.lstm_in_size
        config["model_config"]["lstm_hidden_size"] = args.lstm_hidden_size
        config["model_config"]["channel_1"] = args.channel_1
        config["model_config"]["channel_2"] = args.channel_2
        config["model_config"]["channel_3"] = args.channel_3
        config["model_config"]["use_last_action_reward"] = args.use_last_action_reward
        config["model_config"]["contact"] = args.contact
        config["model_config"]["time_till_end"] = args.time_till_end
        config["model_config"]["one_hot_message"] = args.one_hot_message
        config["model_config"]["actor_hidden_size"] = args.actor_hidden_size
        config["model_config"]["critic_hidden_size"] = args.critic_hidden_size

    return config



def handle_dones(dones):
    """Handles the dones. Converts bools to binary tensor for each agent"""
    dones_out =  {}
    for key in dones.keys():
        if key == "__all__":
            continue
        else:
            #dones_out[key] = torch.tensor(dones[key].astype(float), dtype=torch.float32)
            dones_out[key] = torch.tensor(dones[key], dtype=torch.float32)
    return dones_out



def print_info(storage, total_completed, rewards, stage_success_dict, stages_sampled, coop_stage_success_dict, coop_stages_sampled,
               single_stage_success_dict, single_stages_sampled, epoch, average_reward_dict, best_average_reward_dict, 
               success_rate_dict, best_sucess_rate_dict, success, achieved_goal, achieved_goal_success,
               stages_average_success_rate, coop_stages_average_success_rate, single_stages_average_success_rate, config):
    """Print info for each episode"""
    end_of_episode_info = {}
    print("Epoch: {0}".format(epoch))
    id = 0
    for a in storage.keys():
        reward = rewards[a]
        completed = total_completed[a]
        episodic_reward = reward / completed
        successes = success[a]
        success_rate = successes / completed
        average_reward_dict[a].append(episodic_reward)
        success_rate_dict[a].append(success_rate)

        if epoch > 25:
            average_reward = sum(average_reward_dict[a][-25:]) / 25
            average_success_rate = sum(success_rate_dict[a][-25:]) / 25


        else:
            average_reward = sum(average_reward_dict[a]) / len(average_reward_dict[a])
            average_success_rate = sum(success_rate_dict[a]) / len(success_rate_dict[a])

                   
        print(best_average_reward_dict[a], average_reward)
        if average_reward > best_average_reward_dict[a]:
            best_average_reward_dict[a] = average_reward
 
        if average_success_rate > best_sucess_rate_dict[a]:
            best_sucess_rate_dict[a] = average_success_rate
            
        end_of_episode_info["agent_{0}".format(id)] = {"completed": completed,
                                                     "reward": reward,
                                                     "successes": successes,
                                                     "success_rate": success_rate,
                                                     "rolling_average_reward": average_reward,
                                                     "rolling_average_success_rate": average_success_rate,
                                                     "best_average_reward": best_average_reward_dict[a],
                                                     "best_success_rate": best_sucess_rate_dict[a],
                                                     "achieved_goal": achieved_goal[a],
                                                     "achieved_goal_success": achieved_goal_success[a],}
        
        print("Agent_{0}: Completed {1} Episodes; ".format(id, completed))

        print("Total Reward: {0}; Average Reward This Epoch: {1}; Rolling Average Reward: {2} Best Average Reward: {3}".format(reward, 
                                                                                    episodic_reward, average_reward, best_average_reward_dict[a]))

        print("Successes: {0}; Success Rate: {1}; Rolling Average Sucess Rate: {2}; Best Rolling Average Sucess Rate: {3}".format(successes, success_rate, 
                                                                                                          average_success_rate, best_sucess_rate_dict[a]))
        
        if config["env_config"]["env_name"] in ["MultiAgentLandmarksComm", "MultiAgentLandmarks"]:
            for g in range(achieved_goal[a].shape[0]):
                print("Goal {0}: {1} Achieved; {2} Achieved Successfully".format(g, achieved_goal[a][g].item(), achieved_goal_success[a][g].item()))

        elif config["env_config"]["env_name"] in ["CraftingEnv", "CoopCraftingEnvComm", "CoopCraftingEnv", "TestCraftingEnv"]:
            for s in range(1, config["env_config"]["stages"] + 1):
                stage_successes = stage_success_dict[a]["stage_{0}".format(s)]
                stage_samples = stages_sampled[a]["stage_{0}".format(s)][-1]
                stage_success_rate = stage_successes / (stage_samples + 1e-8)
                
                end_of_episode_info["agent_{0}".format(id)]["stage_{0}_successes".format(s)] = stage_successes
                end_of_episode_info["agent_{0}".format(id)]["stage_{0}_samples".format(s)] = stage_samples
                end_of_episode_info["agent_{0}".format(id)]["stage_{0}_success_rate".format(s)] = stage_success_rate
                
                if config["env_config"]["env_name"] in ["CoopCraftingEnv", "CoopCraftingEnvComm", "TestCraftingEnv"]:
                    coop_stage_successes = coop_stage_success_dict[a]["stage_{0}".format(s)]
                    coop_stage_samples = coop_stages_sampled[a]["stage_{0}".format(s)][-1]
                    coop_stage_success_rate = coop_stage_successes / (coop_stage_samples + 1e-8)
                    end_of_episode_info["agent_{0}".format(id)]["stage_{0}_coop_successes".format(s)] = coop_stage_successes
                    end_of_episode_info["agent_{0}".format(id)]["stage_{0}_coop_samples".format(s)] = coop_stage_samples
                    end_of_episode_info["agent_{0}".format(id)]["stage_{0}_coop_success_rate".format(s)] = coop_stage_success_rate
                    
                    single_stage_successes = single_stage_success_dict[a]["stage_{0}".format(s)]
                    single_stage_samples = single_stages_sampled[a]["stage_{0}".format(s)][-1]
                    single_stage_success_rate = single_stage_successes / (single_stage_samples + 1e-8)
                    end_of_episode_info["agent_{0}".format(id)]["stage_{0}_single_successes".format(s)] = single_stage_successes
                    end_of_episode_info["agent_{0}".format(id)]["stage_{0}_single_samples".format(s)] = single_stage_samples
                    end_of_episode_info["agent_{0}".format(id)]["stage_{0}_single_success_rate".format(s)] = single_stage_success_rate
                             
                    print("Stage {0}: {1} Successes; {2} Samples; {3} Success Rate; Coop Success Rate: {4}; Single Success Rate: {5}".format(s,
                            stage_successes, stage_samples, stage_success_rate, coop_stage_success_rate, single_stage_success_rate))
                
                else:
                     print("Stage {0}: {1} Successes; {2} Samples; {3} Success Rate; ".format(s, 
                        stage_successes, stage_samples, stage_success_rate))
        else:
            pass

                

        id += 1
        
    
    return end_of_episode_info




def record_video(config, env, policy_dict, episodes, video_path, update, test=False):
    """Records a video of the policy in the environment"""
    num_steps = config["env_config"]["timelimit"] * episodes
    frames = []
    infos = []

    if config["env_config"]["env_name"] in COMM_ENVS:
        next_obs, next_messages_in, next_contact, next_time_till_end, _ = env.reset(0)
    else:
        next_obs, next_contact, next_time_till_end, _ = env.reset(0)

    next_dones = {"agent_{0}".format(a): torch.zeros((1, config["env_config"]["num_envs"])) for a in range(config["env_config"]["num_agents"])}
    past_actions = torch.zeros((num_steps, config["env_config"]["num_agents"]) + env.action_space_shape)
    past_rewards = torch.zeros((num_steps, config["env_config"]["num_agents"]))
    next_lstm_state = (torch.zeros(config["model_config"]["lstm_layers"], config["env_config"]["num_agents"], config["model_config"]["lstm_hidden_size"]),
                            torch.zeros(config["model_config"]["lstm_layers"], config["env_config"]["num_agents"], config["model_config"]["lstm_hidden_size"]))
    
    for s in range(num_steps):
        frame = env.vector_env.envs[0].render() * 255.0
        #Convert the frame to uint8 and handle the normalization
        frames.append(frame.astype(np.uint8))

        actions = torch.zeros((1, config["env_config"]["num_agents"]) + env.action_space_shape)
        with torch.no_grad():
            for a in range(config["env_config"]["num_agents"]):
                if config["env_config"]["env_name"] in COMM_ENVS:
                    actions[:,a], _, _, _, next_agent_lstm_state = policy_dict["agent_{0}".format(a)].get_action_and_value(
                        next_obs["agent_{0}".format(a)].reshape((1,) + env.observation_space.shape),
                        (next_lstm_state[0][:,a].unsqueeze(dim=1), next_lstm_state[1][:,a].unsqueeze(dim=1)),
                        next_dones["agent_{0}".format(a)].unsqueeze(dim=0),
                        next_messages_in["agent_{0}".format(a)].reshape((1,) + env.message_shape),
                        past_actions[s, a].unsqueeze(dim=0),
                        past_rewards[s, a].unsqueeze(dim=0).reshape(-1, 1),
                        next_contact["agent_{0}".format(a)],
                        next_time_till_end["agent_{0}".format(a)],
                        )
                else:
                    actions[:,a], _, _, _, next_agent_lstm_state = policy_dict["agent_{0}".format(a)].get_action_and_value(
                        next_obs["agent_{0}".format(a)].reshape((1,) + env.observation_space.shape),
                        (next_lstm_state[0][:,a].unsqueeze(dim=1), next_lstm_state[1][:,a].unsqueeze(dim=1)),
                        next_dones["agent_{0}".format(a)].unsqueeze(dim=0),
                        past_actions[s, a].unsqueeze(dim=0),
                        past_rewards[s, a].unsqueeze(dim=0).reshape(-1, 1),
                        next_contact["agent_{0}".format(a)],
                        next_time_till_end["agent_{0}".format(a)],
                        )
                if s < num_steps - 1:
                    past_actions[s + 1, a] = actions[:,a].squeeze(dim=0)
                next_lstm_state[0][:,a] = next_agent_lstm_state[0].squeeze(dim=1)
                next_lstm_state[1][:,a] = next_agent_lstm_state[1].squeeze(dim=1)

        if config["env_config"]["env_name"] in COMM_ENVS:
            input_dict = {}
            movement_actions = actions[:,:, :env.movement_shape[0]]
            message_actions = actions[:,:, env.movement_shape[0]:]
            input_dict["actions"] = movement_actions
            input_dict["messages"] = message_actions
            next_obs, next_messages_in, rewards ,dones, next_contact, next_time_till_end, info = env.step(input_dict)
        else:
            next_obs, rewards, dones, next_contact, next_time_till_end, info = env.step(actions)
        next_dones = handle_dones(dones)
        for a in range(config["env_config"]["num_agents"]):
            if s < num_steps - 1:
                past_rewards[s + 1, a] = rewards["agent_{0}".format(a)]
            
        infos.append(info)
        if dones["__all__"]:
            if config["env_config"]["env_name"] in COMM_ENVS:
                next_obs, next_messages_in, next_contact, next_time_till_end, _ = env.reset(0)
            else:
                next_obs, next_contact, next_time_till_end, _ = env.reset(0)
    
    #Create video from frames using different library than opencv
    if test:
        writer = imageio.get_writer(video_path + "video_test.mp4".format(update), fps=30, codec="libx264")
    else:
        writer = imageio.get_writer(video_path + "video_epoch_{0}.mp4".format(update), fps=30, codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    save_info_to_csv(infos, video_path + "info_epoch_{0}.csv".format(update))




def flatten_dict(data, parent_key='', sep='.'):
    items = []
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, torch.Tensor):
            # Convert Torch tensor to a NumPy array and then to a normal Python list
            items.append((new_key, v.cpu().numpy().tolist()))
        else:
            items.append((new_key, v))
    return dict(items)

def save_info_to_csv(infos, save_path):
    # Flatten the nested dictionaries in each info dictionary
    flattened_infos = [flatten_dict(info) for info in infos]

    # Extract the keys from the flattened dictionary
    keys = list(flattened_infos[0].keys())

    # Open the CSV file in write mode
    with open(save_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)

        # Write the header row
        writer.writeheader()

        # Write each info dictionary as a row in the CSV file
        for info in flattened_infos:
            writer.writerow(info)
