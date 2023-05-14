import torch
import yaml

def build_storage(config, env):
    """Builds a dict for the storage of the rollout data for each policy"""
    num_steps = config["rollout_steps"]
    num_envs = config["env_config"]["num_envs"]
    device = config["device"]
    num_layers = config["model_config"]["lstm_layers"]
    hidden_size = config["model_config"]["lstm_hidden_size"]

    storage = {}
    for a in range(config["env_config"]["num_agents"]):
        storage["agent_{0}".format(a)] = {
                    "obs": torch.zeros((1, num_envs) + env.observation_space.shape).to(device),
                    "dones":torch.zeros((1, num_envs)).to(device),
                    "is_training":torch.zeros((1, num_envs)).to(device),
                    "collected_env_steps":torch.zeros((num_envs)).to(device),
                    "next_lstm_state":(
                                        torch.zeros(num_layers, num_envs, hidden_size).to(device),
                                        torch.zeros(num_layers, num_envs, hidden_size).to(device),
                                        ),
                    "rewards":torch.zeros((1, num_envs)).to(device),
                    "actions":torch.zeros((1, num_envs) + env.action_space.shape).to(device),
                    "values":torch.zeros((1, num_envs)).to(device),
                    "logprobs":torch.zeros((1, num_envs)).to(device),
                    "returns":torch.zeros((num_steps, num_envs)).to(device),
                    "advantages":torch.zeros((num_steps, num_envs)).to(device),
                    }
              
    return storage

def get_init_tensors(storage):
    action = torch.zeros_like(storage["actions"])
    logprob = torch.zeros_like(storage["logprobs"])
    value = torch.zeros_like(storage["values"])
    next_lstm_state = (torch.zeros_like(storage["next_lstm_state"][0], torch.zeros_like(storage["next_lstm_state"][1])))
    return action, logprob, value, next_lstm_state


def truncate_storage(storage, config):
    """Get the actual training data for each environment and truncate to steps_per_env"""
    rollout_steps = config["rollout_steps"]
    num_envs = config["env_config"]["num_envs"]
    steps_per_env = int(rollout_steps // num_envs)

    for agent in storage.keys():
        for key in storage[agent].keys():
            if key in ["obs", "dones", "rewards", "actions", "values", "logprobs"]:
                storage[agent][key] = storage[agent][key][storage[agent]["is_training"].bool()][:steps_per_env]
            else:
                continue
    return storage

def reset_storage(storage, config, env):
    """Reset the storage dict to all zeros. LSTM state is not reset across epochs!!"""
    num_steps = config["rollout_steps"]
    num_envs = config["env_config"]["num_envs"]
    device = config["device"]

    for agent in storage.keys():
        for key in storage[agent].keys():
            if key in ["obs"]:
                storage[agent][key] = torch.zeros((1, num_envs) + env.observation_space.shape).to(device)
            elif key in ["dones", "is_training", "rewards", "actions", "values", "logprobs"]:
                storage[agent][key] = torch.zeros((1, num_envs)).to(device)
            elif key in ["collected_env_steps"]:
                storage[agent][key] = torch.zeros((num_envs)).to(device)
            elif key in ["returns", "advantages"]:
                storage[agent][key] = torch.zeros((num_steps, num_envs)).to(device)
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

        config["total_steps"] = args.total_steps
        config["rollout_steps"] = args.rollout_steps
        config["batch_size"] = args.batch_size
        config["seed"] = args.seed
        config["anneal_lr"] = args.anneal_lr
        config["lr"] = args.lr
        config["gae"] = args.gae
        config["gae_lambda"] = args.gae_lambda
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

        config["model_config"] = {}
        config["model_config"]["lstm_layers"] = args.lstm_layers
        config["model_config"]["lstm_in_size"] = args.lstm_in_size
        config["model_config"]["lstm_hidden_size"] = args.lstm_hidden_size
        config["model_config"]["channel_1"] = args.channel_1
        config["model_config"]["channel_2"] = args.channel_2
        config["model_config"]["channel_3"] = args.channel_3

    return config

def handle_dones(dones):
    """Handles the dones. Converts bools to binary tensor for each agent"""
    dones_out =  {}
    for key in dones.keys():
        if key == "__all__":
            continue
        else:
            dones_out[key] = torch.tensor(dones[key].astype(float), dtype=torch.float32)
    return dones_out


def print_info(storage, epoch):
    """Print info for each episode"""
    end_of_episode_info = {}
    print("Epoch: {0}".format(epoch))
    for a in range(len(storage["agent_0"]["rewards"])):
        completed = sum(storage["agent_0"]["dones"][a].cpu().numpy().astype(float))
        reward = sum(storage["agent_0"]["rewards"][a].cpu().numpy().astype(float))
        success_rate = completed / reward
        end_of_episode_info["agent_{0}".format*a] = {"completed": completed,
                                                     "reward": reward,
                                                     "success_rate": success_rate}
        
        
        print("Agent_{0}: Completed {1} Episodes; Total Reward: {2}; Success Rate: {3}".format(a, completed, reward, success_rate))
    return end_of_episode_info

