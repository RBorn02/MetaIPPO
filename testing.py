import torch
import random
import os
import numpy as np


from distutils.util import strtobool
from argparse import ArgumentParser

from agent import LSTMAgent, CommsLSTMAgent
from policy import LSTM_PPO_Policy
from mp_train import rollout
from Envs.environment_handler import EnvironmentHandler
from Utils.train_utils import record_video, build_config, build_storage, handle_dones, print_info

parser = ArgumentParser()

parser.add_argument("--pretrained", default=None, type=str,
                    help="Path to the pretrained models")
parser.add_argument("--config", default=None,
                    help="Optional path to the config yaml")
parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                    help="Toggles debug mode, disables logging")
parser.add_argument("--env_name", type=str, default="MultiAgentLandmarks",
                    help="Name of the environment to use")
parser.add_argument("--num_agents", type=int, default=2,
                    help="Number of agents in the environment")
parser.add_argument("--num_landmarks", type=int, default=4,
                    help="Number of landmarks in the environment")
parser.add_argument("--random_assign", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="If true, changes the shape and color of the landmarks randomly")
parser.add_argument("--message_length", type=int, default=1,
                    help="Length of the message")
parser.add_argument("--vocab_size", type=int, default=3,
                    help="Size of the vocabulary")
parser.add_argument("--min_prob", type=float, default=0.025,
                    help="Min probability of a stage being sampled in crafting env")
parser.add_argument("--max_prob", type=float, default=0.95,
                    help="Max probability of a stage being sampled in crafting env")
parser.add_argument("--playground_height", type=int, default=300,
                    help="Height of the playground")
parser.add_argument("--playground_width", type=int, default=300,
                    help="Width of the playground")
parser.add_argument("--num_envs", type=int, default=16,
                    help="Number of environments to vectorize")
parser.add_argument("--time_limit", type=int, default=250,
                    help="Number of max steps per episode")
parser.add_argument("--coop_chance", type=float, default=1.0,
                    help="Chance of cooperative goal")
parser.add_argument("--single_goal", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="Only sample a goal once per episode")
parser.add_argument("--single_reward", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="Controls wether agents are done after receiving the reward or continue getting rewards")
parser.add_argument("--total_steps", type=int, default=2.5*10e7,
                    help="Number of steps to train for")
parser.add_argument("--rollout_steps", type=int, default=32000,
                    help="Number of steps per rollout")
parser.add_argument("--seed", type=int, default=1,
                    help="Random seed")
parser.add_argument("--device", type=str, default="cpu",
                    help="Device to use for training")
parser.add_argument("--num_workers", type=int, default=8,
                    help="Number of workers to use for training")

# PPO specific arguments
parser.add_argument("--anneal_lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="Toggle learning rate annealing for policy and value networks")
parser.add_argument("--lr", type=float, default=2.5e-4,
                    help="Learning rate")
parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="Use GAE for advantage computation")
parser.add_argument("--gamma", type=float, default=0.99,
                    help="the discount factor gamma")
parser.add_argument("--gae_lambda", type=float, default=1.0,
                    help="the lambda for the general advantage estimation")
parser.add_argument("--num_minibatches", type=int, default=4,
                    help="the number of mini-batches")
parser.add_argument("--update_epochs", type=int, default=4,
                    help="the K epochs to update the policy")
parser.add_argument("--norm_adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="Toggles advantages normalization")
parser.add_argument("--clip_coef", type=float, default=0.1,
                    help="the surrogate clipping coefficient")
parser.add_argument("--clip_vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
parser.add_argument("--ent_coef", type=float, default=0.001,
                    help="coefficient of the entropy")
parser.add_argument("--vf_coef", type=float, default=1.0,
                    help="coefficient of the value function")
parser.add_argument("--max_grad_norm", type=float, default=0.5,
                    help="the maximum norm for the gradient clipping")
parser.add_argument("--target_kl", type=float, default=None,
                    help="the target KL divergence threshold")
parser.add_argument("--record_video_every", type=int, default=10,
                    help="Record a video every n episodes")

# Model Specific arguments
parser.add_argument("--channel_1", type=int, default=16,
                    help="Number of channels in the first convolutional layer")
parser.add_argument("--channel_2", type=int, default=32,
                    help="Number of channels in the second convolutional layer")
parser.add_argument("--channel_3", type=int, default=32,
                    help="Number of channels in the third convolutional layer")
parser.add_argument("--lstm_in_size", type=int, default=64,
                    help="Size of the LSTM hidden state")
parser.add_argument("--lstm_hidden_size", type=int, default=64,
                    help="Size of the LSTM hidden state")
parser.add_argument("--lstm_layers", type=int, default=1,
                    help="Number of LSTM layers")
parser.add_argument("--use_last_action_reward", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="Toggles whether or not to use the last action and reward as input to the LSTM")
parser.add_argument("--contact", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="Toggles whether or not to use contact information as input to the LSTM")
parser.add_argument("--one_hot_message", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                    help="Toggles whether or not to use one hot encoding for the message")

args = parser.parse_args()
args.batch_size = int(args.rollout_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)

config = build_config(args)
config["env_config"]["test"] = True
device = config["device"]

#Build the environemnt
env = EnvironmentHandler(config)

#Build the agents with their corresponding optimizers
if config["env_config"]["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm", "TreasureHuntComm"]:
    agent_dict = {"agent_{0}".format(a): CommsLSTMAgent(env, config).share_memory().to(device) for a in range(config["env_config"]["num_agents"])}
else:
    agent_dict = {"agent_{0}".format(a): LSTMAgent(env, config).share_memory().to(device) for a in range(config["env_config"]["num_agents"])}
optimizer_dict = {"agent_{0}".format(a): {"optimizer": torch.optim.Adam(agent_dict["agent_{0}".format(a)].parameters(),
                                                                        config["lr"], eps=1e-5)} 
                        for a in range(config["env_config"]["num_agents"])}

#Load the pretrained models
for a in range(config["env_config"]["num_agents"]):
    agent_dict["agent_{0}".format(a)].load_state_dict(torch.load(os.path.join(config["pretrained"], "agent_{0}_model.pt".format(a)))["model"])
    optimizer_dict["agent_{0}".format(a)]["optimizer"].load_state_dict(torch.load(os.path.join(config["pretrained"], "agent_{0}_model.pt".format(a)))["optimizer"])
    print("Loaded pretrained model for agent {0}".format(a))

    #Initiate the learning rate if pretrained
    lrnow = optimizer_dict["agent_0"]["optimizer"].param_groups[0]['lr']

#Build the policies
policy_dict = {"agent_{0}".format(a): LSTM_PPO_Policy(config, agent_dict["agent_{0}".format(a)], optimizer_dict["agent_{0}".format(a)]["optimizer"]) 
               for a in range(config["env_config"]["num_agents"])}

if __name__ == "__main__":

    #Random seeding
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True

    storage = build_storage(config, env)

    training_info = {}
    average_reward = {"agent_{0}".format(a): [] for a in range(config["env_config"]["num_agents"])}
    best_average_reward = {"agent_{0}".format(a): 0.0 for a in range(config["env_config"]["num_agents"])}
    average_success_rate = {"agent_{0}".format(a): [] for a in range(config["env_config"]["num_agents"])}
    best_average_success_rate = {"agent_{0}".format(a): 0.0 for a in range(config["env_config"]["num_agents"])}
    prev_best = 0.0

    stages_rolling_success_rate = {"agent_{0}".format(a): {"stage_{0}".format(s): [] for s in range(1, 4)} 
                                    for a in range(config["env_config"]["num_agents"])}

    num_steps = config["rollout_steps"] // config["env_config"]["num_envs"]

    if config["env_config"]["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm", "TreasureHuntComm"]:
            next_obs, next_messages_in, next_contact, _ = env.reset_all([i for i in range(config["env_config"]["num_envs"])])
    else:
        next_obs, next_contact, _ = env.reset_all([i for i in range(config["env_config"]["num_envs"])])


    next_dones = {"agent_{0}".format(a): torch.zeros((1, config["env_config"]["num_envs"])).to(device) for a in range(config["env_config"]["num_agents"])}
    last_actions = {"agent_{0}".format(a): storage["agent_{0}".format(a)]["actions"][0].to(device) for a in range(config["env_config"]["num_agents"])}
    last_rewards = {"agent_{0}".format(a): storage["agent_{0}".format(a)]["rewards"][0].to(device) for a in range(config["env_config"]["num_agents"])}
    
    success_rate = {}
    stages_success_info = {}
    achieved_goal = {}
    achieved_goal_success = {}
    for a in range(config["env_config"]["num_agents"]):
            storage["agent_{0}".format(a)]["initial_lstm_state"] = (storage["agent_{0}".format(a)]["next_lstm_state"][0].clone(),
                                                                    storage["agent_{0}".format(a)]["next_lstm_state"][1].clone())
            success_rate["agent_{0}".format(a)] = 0
            achieved_goal["agent_{0}".format(a)] = torch.zeros((config["env_config"]["num_landmarks"])) 
            achieved_goal_success["agent_{0}".format(a)] = torch.zeros((config["env_config"]["num_landmarks"]))
            stages_success_info["agent_{0}".format(a)] = {"stage_{0}".format(s): (0, 0) for s in range(1, 4)} 

    rollout_step = 0

    while rollout_step < num_steps:
        for a in range(config["env_config"]["num_agents"]):
            next_agent_obs = next_obs["agent_{0}".format(a)].to(device)
            next_agent_dones = next_dones["agent_{0}".format(a)].to(device)
            next_agent_lstm_state = storage["agent_{0}".format(a)]["next_lstm_state"]
            next_agent_contact = next_contact["agent_{0}".format(a)].to(device)
            last_agent_actions = last_actions["agent_{0}".format(a)]
            last_agent_rewards = last_rewards["agent_{0}".format(a)]
            
            storage["agent_{0}".format(a)]["obs"][rollout_step] = next_agent_obs
            storage["agent_{0}".format(a)]["dones"][rollout_step] = next_agent_dones
            storage["agent_{0}".format(a)]["contact"][rollout_step] = next_agent_contact
            storage["agent_{0}".format(a)]["last_actions"][rollout_step] = last_agent_actions
            storage["agent_{0}".format(a)]["last_rewards"][rollout_step] = last_agent_rewards

            if config["env_config"]["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm", "TreasureHuntComm"]:
                next_agent_message_in = next_messages_in["agent_{0}".format(a)].to(device)
                storage["agent_{0}".format(a)]["message_in"][rollout_step] = next_agent_message_in
        
            #Get the actions from the policy
            with torch.no_grad():
                if config["env_config"]["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm", "TreasureHuntComm"]:
                    
                    action, log_prob, _, value, next_agent_lstm_state = policy_dict["agent_{0}".format(a)].get_action_and_value(
                                                                                                            next_agent_obs,
                                                                                                            next_agent_lstm_state,
                                                                                                            next_agent_dones,
                                                                                                            last_agent_actions,
                                                                                                            last_agent_rewards.unsqueeze(dim=1),
                                                                                                            next_agent_contact.transpose(0,1),
                                                                                                            next_agent_message_in.squeeze(dim=0))
                    
                else:
                    action, log_prob, _, value, next_agent_lstm_state = policy_dict["agent_{0}".format(a)].get_action_and_value(
                                                                                                            next_agent_obs,
                                                                                                            next_agent_lstm_state,
                                                                                                            next_agent_dones,
                                                                                                            last_agent_actions,
                                                                                                            last_agent_rewards.unsqueeze(dim=1),
                                                                                                            next_agent_contact.transpose(0,1))
                
            storage["agent_{0}".format(a)]["values"][rollout_step] = value.transpose(0, 1)
            storage["agent_{0}".format(a)]["actions"][rollout_step] = action
            storage["agent_{0}".format(a)]["logprobs"][rollout_step] = log_prob
            storage["agent_{0}".format(a)]["next_lstm_state"] = (next_agent_lstm_state[0], next_agent_lstm_state[1])


        #Take step in environment
        if config["env_config"]["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm", "TreasureHuntComm"]:
                    input_dict = {}
                    actions = torch.cat([storage["agent_{0}".format(a)]["actions"][rollout_step][:,:-config["env_config"]["message_length"]].unsqueeze(dim=1)
                                    for a in range(config["env_config"]["num_agents"])], dim=1)
                    messages = torch.cat([storage["agent_{0}".format(a)]["actions"][rollout_step][:,-config["env_config"]["message_length"]:].unsqueeze(dim=1)
                                    for a in range(config["env_config"]["num_agents"])], dim=1)

                    input_dict["actions"] = actions.cpu()
                    input_dict["messages"] = messages.cpu()

                    next_obs, next_messages_in, rewards, dones, next_contact, infos = env.step(input_dict)
                
        else:
            actions = torch.cat([storage["agent_{0}".format(a)]["actions"][rollout_step].unsqueeze(dim=1)
                            for a in range(config["env_config"]["num_agents"])], dim=1)

            next_obs, rewards, dones, next_contact, infos = env.step(actions.cpu())


        
        #Handle the dones and convert the bools to binary tensors
        next_dones = handle_dones(dones)
        
        
        #Store the rewards, success rate, goal line and handle past actions and rewards
        for a in range(config["env_config"]["num_agents"]):
            storage["agent_{0}".format(a)]["rewards"][rollout_step] = rewards["agent_{0}".format(a)].to(device)
            
            last_actions["agent_{0}".format(a)] = storage["agent_{0}".format(a)]["actions"][rollout_step].to(device)
            last_rewards["agent_{0}".format(a)] = storage["agent_{0}".format(a)]["rewards"][rollout_step].to(device)

            success_rate["agent_{0}".format(a)] += torch.sum(infos["agent_{0}".format(a)]["success"]).item()

            if config["env_config"]["env_name"] in ["CraftingEnv", "CraftingEnvComm"]:
                for s in range(1, 4):
                    num_stage_sampled = torch.sum(torch.where(infos["agent_{0}".format(a)]["success_stage_{0}".format(s)] >= 0, 1.0, 0.0)).item()
                    num_stage_success = torch.sum(torch.where(infos["agent_{0}".format(a)]["success_stage_{0}".format(s)] == 1, 1.0, 0.0)).item()
                    prev_stage_success = stages_success_info["agent_{0}".format(a)]["stage_{0}".format(s)][1]
                    stages_success_info["agent_{0}".format(a)]["stage_{0}".format(s)] = (num_stage_sampled, num_stage_success + prev_stage_success)


            for e in range(config["env_config"]["num_envs"]):
                idx = infos["agent_{0}".format(a)]["goal_line"][0][e].squeeze()
                if idx >= 0:
                    achieved_goal["agent_{0}".format(a)][int(idx)] += 1.0
                    if infos["agent_{0}".format(a)]["success"][0][e]:
                        achieved_goal_success["agent_{0}".format(a)][int(idx)] += 1.0
        
        
        #Reset Environments that are done
        for e in range(config["env_config"]["num_envs"]):
            if dones["__all__"][e]:
                for a in range(config["env_config"]["num_agents"]):
                    if config["env_config"]["env_name"] in ["MultiAgentLandmarksComm", "LinRoomEnvComm", "LinLandmarksEnvComm", "TreasureHuntComm"]:
                        reset_obs, reset_messages, reset_contact, _ = env.reset(e)
                        next_obs["agent_{0}".format(a)][0][e] = reset_obs["agent_{0}".format(a)].to(device)
                        next_messages_in["agent_{0}".format(a)][0][e] = reset_messages["agent_{0}".format(a)].to(device)
                        next_contact["agent_{0}".format(a)][0][e] = reset_contact["agent_{0}".format(a)].to(device)
                    else:
                        reset_obs, reset_contact, _ = env.reset(e)
                        next_obs["agent_{0}".format(a)][0][e] = reset_obs["agent_{0}".format(a)].to(device)
                        next_contact["agent_{0}".format(a)][0][e] = reset_contact["agent_{0}".format(a)].to(device)
                
                #successes_this_episode = achieved_goal_success["agent_0"].sum().item()
                #print(successes_this_episode)
                

        rollout_step += 1

    #Print info
    print_info(storage, next_dones, 0, average_reward, best_average_reward,
                average_success_rate, best_average_success_rate, success_rate,
                achieved_goal, achieved_goal_success, stages_success_info, stages_rolling_success_rate, config)

    #Record a video
    path = config["pretrained"].removesuffix("models")
    video_path = os.path.join(path, "Videos/")

    video_config = {}
    video_config["env_config"] = config["env_config"].copy()
    video_config["env_config"]["num_envs"] = 1
    video_config["model_config"] = config["model_config"].copy()
    video_env = EnvironmentHandler(video_config)
    record_video(video_config, video_env, policy_dict, 20, video_path, 0, True)

    print("Finished testing")
    
