import torch
import yaml

def build_storage(config, env):
    """Builds a dict for the storage of the rollout data for each policy"""
    num_steps = config["rollout_steps"] // (config["num_workers"] * config["env_config"]["num_agents"])
    num_envs = config["env_config"]["num_envs"]
    device = config["device"]
    num_layers = config["model_config"]["lstm_layers"]
    hidden_size = config["model_config"]["lstm_hidden_size"]

    storage = {}
    for a in range(config["env_config"]["num_agents"]):
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
                    "actions":torch.zeros((num_steps, num_envs) + env.action_space.shape).to(device),
                    "last_actions":torch.zeros((num_steps, num_envs) + env.action_space.shape).to(device),
                    "values":torch.zeros((num_steps, num_envs)).to(device),
                    "logprobs":torch.zeros((num_steps, num_envs)).to(device),
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
    success_rate = {}
    achieved_goal = {}
    achieved_goal_success = {}
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
        storage_out[agent]["values"] = torch.cat([batch[i][0][agent]["values"] for i in range(len(batch))], dim=1)
        storage_out[agent]["logprobs"] = torch.cat([batch[i][0][agent]["logprobs"] for i in range(len(batch))], dim=1)

        next_obs[agent] = torch.cat([batch[i][1][agent] for i in range(len(batch))], dim=1)
        next_dones[agent] = torch.cat([batch[i][2][agent] for i in range(len(batch))], dim=1)
        success_rate[agent] = sum([batch[i][3][agent] for i in range(len(batch))])
        achieved_goal[agent] = torch.sum(torch.cat([batch[i][4][agent].unsqueeze(dim=0) for i in range(len(batch))], dim=0), dim=0)
        achieved_goal_success[agent] = torch.sum(torch.cat([batch[i][5][agent].unsqueeze(dim=0) for i in range(len(batch))], dim=0), dim=0)
    
    return storage_out, next_obs, next_dones, success_rate, achieved_goal, achieved_goal_success
        
    

def reset_storage(storage, config, env):
    """Reset the storage dict to all zeros. LSTM state is not reset across epochs!!"""
    num_steps = config["rollout_steps"] // (config["num_workers"] * config["env_config"]["num_agents"])
    num_envs = config["env_config"]["num_envs"]
    device = config["device"]

    for agent in storage.keys():
        for key in storage[agent].keys():
            if key in ["obs"]:
                storage[agent][key] = torch.zeros((num_steps, num_envs) + env.observation_space.shape).to(device)
            elif key in ["dones", "rewards", "values", "logprobs"]:
                storage[agent][key] = torch.zeros((num_steps, num_envs)).to(device)
            elif key in ["actions"]:
                storage[agent][key] = torch.zeros((num_steps, num_envs) + env.action_space.shape).to(device)
            else:
                continue
    return storage
            


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

        config["model_config"] = {}
        config["model_config"]["lstm_layers"] = args.lstm_layers
        config["model_config"]["lstm_in_size"] = args.lstm_in_size
        config["model_config"]["lstm_hidden_size"] = args.lstm_hidden_size
        config["model_config"]["channel_1"] = args.channel_1
        config["model_config"]["channel_2"] = args.channel_2
        config["model_config"]["channel_3"] = args.channel_3
        config["model_config"]["use_last_action_reward"] = args.use_last_action_reward

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


def print_info(storage, next_dones, epoch, average_reward_dict, best_average_reward_dict, 
               success_rate_dict, best_sucess_rate_dict, success, achieved_goal, achieved_goal_success):
    """Print info for each episode"""
    end_of_episode_info = {}
    print("Epoch: {0}".format(epoch))
    id = 0
    for a in storage.keys():
        #print(storage[a]["rewards"])
        completed = torch.sum(torch.cat((storage[a]["dones"][1:].cpu(), next_dones[a]), dim=0))
        reward = torch.sum(storage[a]["rewards"].cpu())
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
        if average_reward > best_average_reward_dict[a]:
            best_average_reward_dict[a] = average_reward
 
        if average_success_rate > best_sucess_rate_dict[a]:
            best_sucess_rate_dict[a] = average_success_rate
            
        end_of_episode_info["agent_{0}".format(id)] = {"completed": completed,
                                                     "reward": reward,
                                                     "average_reward": episodic_reward,
                                                     "average_success_rate": average_reward,
                                                     "successes": successes,
                                                     "success_rate": success_rate,
                                                     "rolling_average_reward": average_reward,
                                                     "rolling_average_success_rate": average_success_rate,
                                                     "best_average_reward": best_average_reward_dict[a],
                                                     "best_success_rate": best_sucess_rate_dict[a],
                                                     "achieved_goal": achieved_goal[a],
                                                     "achieved_goal_success": achieved_goal_success[a]}
        
        print("Agent_{0}: Completed {1} Episodes; ".format(id, completed))

        print("Total Reward: {0}; Average Reward This Epoch: {1}; Rolling Average Reward: {2} Best Average Reward: {3}".format(reward, 
                                                                                    episodic_reward, average_reward, best_average_reward_dict[a]))

        print("Successes: {0}; Success Rate: {1}; Rolling Average Sucess Rate: {2}; Best Rolling Average Sucess Rate: {3}".format(successes, success_rate, 
                                                                                                            average_success_rate, best_sucess_rate_dict[a]))
        for g in range(achieved_goal[a].shape[0]):
            print("Goal {0}: {1} Achieved; {2} Achieved Successfully".format(g, achieved_goal[a][g].item(), achieved_goal_success[a][g].item()))
        
        id += 1
        
    
    return end_of_episode_info



        
    
