import os
import random
import torch
import torch.multiprocessing as mp
import numpy as np
from distutils.util import strtobool
import time
import traceback
import wandb

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from policy import LSTM_PPO_Policy
from agent import LSTMAgent
from Envs.environment_handler import EnvironmentHandler
from Utils.train_utils import build_config, build_storage, handle_dones, print_info, get_init_tensors, truncate_storage, reset_storage, build_storage_from_batch

parser = ArgumentParser()

parser.add_argument("--config", default=None,
                    help="Optional path to the config yaml")
parser.add_argument("--pretrained", default=None, type=str,
                    help="Optional path to the pretrained models")
parser.add_argument("--env_name", type=str, default="MultiAgentLandmarks",
                    help="Name of the environment to use")
parser.add_argument("--num_agents", type=int, default=2,
                    help="Number of agents in the environment")
parser.add_argument("--num_landmarks", type=int, default=3,
                    help="Number of landmarks in the environment")
parser.add_argument("--num_envs", type=int, default=8,
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
parser.add_argument("--rollout_steps", type=int, default=16000,
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
parser.add_argument("--gae_lambda", type=float, default=0.95,
                    help="the lambda for the general advantage estimation")
parser.add_argument("--num_minibatches", type=int, default=8,
                    help="the number of mini-batches")
parser.add_argument("--update_epochs", type=int, default=4,
                    help="the K epochs to update the policy")
parser.add_argument("--norm_adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="Toggles advantages normalization")
parser.add_argument("--clip_coef", type=float, default=0.1,
                    help="the surrogate clipping coefficient")
parser.add_argument("--clip_vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
parser.add_argument("--ent_coef", type=float, default=0.01,
                    help="coefficient of the entropy")
parser.add_argument("--vf_coef", type=float, default=0.5,
                    help="coefficient of the value function")
parser.add_argument("--max_grad_norm", type=float, default=0.5,
                    help="the maximum norm for the gradient clipping")
parser.add_argument("--target_kl", type=float, default=None,
                    help="the target KL divergence threshold")

# Model Specific arguments
parser.add_argument("--channel_1", type=int, default=32,
                    help="Number of channels in the first convolutional layer")
parser.add_argument("--channel_2", type=int, default=64,
                    help="Number of channels in the second convolutional layer")
parser.add_argument("--channel_3", type=int, default=64,
                    help="Number of channels in the third convolutional layer")
parser.add_argument("--lstm_in_size", type=int, default=256,
                    help="Size of the LSTM hidden state")
parser.add_argument("--lstm_hidden_size", type=int, default=256,
                    help="Size of the LSTM hidden state")
parser.add_argument("--lstm_layers", type=int, default=1,
                    help="Number of LSTM layers")
parser.add_argument("--use_last_action_reward", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="Toggles whether or not to use the last action and reward as input to the LSTM")




args = parser.parse_args()
args.batch_size = int(args.rollout_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)

config = build_config(args)
device = config["device"]

#Build the environemnt
env = EnvironmentHandler(config)

#Build the agents with their corresponding optimizers
agent_dict = {"agent_{0}".format(a): LSTMAgent(env, config).share_memory().to(device) for a in range(config["env_config"]["num_agents"])}
optimizer_dict = {"agent_{0}".format(a): {"optimizer": torch.optim.Adam(agent_dict["agent_{0}".format(a)].parameters(),
                                                                        config["lr"], eps=1e-5)} 
                        for a in range(config["env_config"]["num_agents"])}

if config["pretrained"] is not None:
    for a in range(config["env_config"]["num_agents"]):
        agent_dict["agent_{0}".format(a)].load_state_dict(torch.load(os.path.join(config["pretrained"], "agent_{0}_model.pt".format(a)))["model"])
        optimizer_dict["agent_{0}".format(a)]["optimizer"].load_state_dict(torch.load(os.path.join(config["pretrained"], "agent_{0}_model.pt".format(a)))["optimizer"])
        print("Loaded pretrained model for agent {0}".format(a))

#Build the policies
policy_dict = {"agent_{0}".format(a): LSTM_PPO_Policy(config, agent_dict["agent_{0}".format(a)], optimizer_dict["agent_{0}".format(a)]["optimizer"]) 
               for a in range(config["env_config"]["num_agents"])}



def rollout(pid, policy_dict, train_queue, done, config):
    device = config["device"]
    env = EnvironmentHandler(config)
    storage = build_storage(config, env)

    next_obs, _ = env.reset_all([i for i in range(config["env_config"]["num_envs"])])
    next_dones = {"agent_{0}".format(a): torch.zeros((1, config["env_config"]["num_envs"])).to(device) for a in range(config["env_config"]["num_agents"])}
    last_actions = {"agent_{0}".format(a): storage["agent_{0}".format(a)]["actions"][0].to(device) for a in range(config["env_config"]["num_agents"])}
    last_rewards = {"agent_{0}".format(a): storage["agent_{0}".format(a)]["rewards"][0].to(device) for a in range(config["env_config"]["num_agents"])}
    
    success_rate = {}
    achieved_goal = {}
    achieved_goal_success = {}
    for a in range(config["env_config"]["num_agents"]):
            storage["agent_{0}".format(a)]["initial_lstm_state"] = (storage["agent_{0}".format(a)]["next_lstm_state"][0].clone(),
                                                                    storage["agent_{0}".format(a)]["next_lstm_state"][1].clone())
            success_rate["agent_{0}".format(a)] = 0
            achieved_goal["agent_{0}".format(a)] = torch.zeros((config["env_config"]["num_landmarks"])) 
            achieved_goal_success["agent_{0}".format(a)] = torch.zeros((config["env_config"]["num_landmarks"])) 

    rollout_step = 0        
    while True:
        try:
            if bool(done[pid]) is False:
                for a in range(config["env_config"]["num_agents"]):
                    next_agent_obs = next_obs["agent_{0}".format(a)].to(device)
                    next_agent_dones = next_dones["agent_{0}".format(a)].to(device)
                    next_agent_lstm_state = storage["agent_{0}".format(a)]["next_lstm_state"]
                    last_agent_actions = last_actions["agent_{0}".format(a)]
                    last_agent_rewards = last_rewards["agent_{0}".format(a)]
                    
                    storage["agent_{0}".format(a)]["obs"][rollout_step] = next_agent_obs
                    storage["agent_{0}".format(a)]["dones"][rollout_step] = next_agent_dones
                    storage["agent_{0}".format(a)]["last_actions"][rollout_step] = last_agent_actions
                    storage["agent_{0}".format(a)]["last_rewards"][rollout_step] = last_agent_rewards
                
                    #Get the actions from the policy
                    with torch.no_grad():
                        action, log_prob, _, value, next_agent_lstm_state = policy_dict["agent_{0}".format(a)].get_action_and_value(
                                                                                                                    next_agent_obs,
                                                                                                                    next_agent_lstm_state,
                                                                                                                    next_agent_dones,
                                                                                                                    last_agent_actions,
                                                                                                                    last_agent_rewards.unsqueeze(dim=1))
                    
                    storage["agent_{0}".format(a)]["values"][rollout_step] = value.transpose(0, 1)
                    storage["agent_{0}".format(a)]["actions"][rollout_step] = action
                    storage["agent_{0}".format(a)]["logprobs"][rollout_step] = log_prob
                    storage["agent_{0}".format(a)]["next_lstm_state"] = (next_agent_lstm_state[0], next_agent_lstm_state[1])

                #Take a step in the environment
                actions = torch.cat([storage["agent_{0}".format(a)]["actions"][rollout_step].unsqueeze(dim=1)
                                    for a in range(config["env_config"]["num_agents"])], dim=1)
        
                next_obs, rewards, dones, infos = env.step(actions.cpu())

                
                #Handle the dones and convert the bools to binary tensors
                next_dones = handle_dones(dones)
                
                #Store the rewards, success rate, goal line and handle past actions and rewards
                for a in range(config["env_config"]["num_agents"]):
                    storage["agent_{0}".format(a)]["rewards"][rollout_step] = rewards["agent_{0}".format(a)].to(device)
                    
                    last_actions["agent_{0}".format(a)] = storage["agent_{0}".format(a)]["actions"][rollout_step].to(device)
                    last_rewards["agent_{0}".format(a)] = storage["agent_{0}".format(a)]["rewards"][rollout_step].to(device)

                    success_rate["agent_{0}".format(a)] += torch.sum(infos["agent_{0}".format(a)]["success"]).item()
                    for e in range(config["env_config"]["num_envs"]):
                        idx = infos["agent_{0}".format(a)]["goal_line"][0][e].squeeze()
                        if idx >= 0:
                            achieved_goal["agent_{0}".format(a)][int(idx)] += 1.0
                            if infos["agent_{0}".format(a)]["success"][0][e]:
                                achieved_goal_success["agent_{0}".format(a)][int(idx)] += 1.0
                
                
                #Reset Environments that are done
                for e in range(config["env_config"]["num_envs"]):
                    if dones["__all__"][e]:
                        reset_obs , _ = env.reset(e)
                        for a in range(config["env_config"]["num_agents"]):
                            next_obs["agent_{0}".format(a)][0][e] = reset_obs["agent_{0}".format(a)].to(device)

                
                #Hold training for the worker if enough data is collected and put it into the training queue
                if rollout_step >= config["rollout_steps"] / (config["num_workers"]*config["env_config"]["num_envs"]):
                    train_queue.put((storage, next_obs, next_dones, success_rate, achieved_goal, achieved_goal_success), block=True)
                    done[pid] = 1
                    rollout_step = 0
                    #Last lstm state is the initial lstm state for the next rollout
                    for a in range(config["env_config"]["num_agents"]):
                        storage["agent_{0}".format(a)]["initial_lstm_state"] = (storage["agent_{0}".format(a)]["next_lstm_state"][0].clone(), 
                                                                                storage["agent_{0}".format(a)]["next_lstm_state"][1].clone())
                        success_rate["agent_{0}".format(a)] = 0.0
                        achieved_goal["agent_{0}".format(a)] = torch.zeros((config["env_config"]["num_landmarks"])) 
                        achieved_goal_success["agent_{0}".format(a)] = torch.zeros((config["env_config"]["num_landmarks"]))
                    
                    print("Worker {0} finished collecting data".format(pid))
                else:
                    rollout_step += 1
                    continue
            
            else:
                time.sleep(1)

        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            break
    

#Train the agents
if __name__ == "__main__":
    #Multi Processing
    os.environ['OMP_NUM_THREADS'] = '1'
    manager = mp.Manager()
    train_queue = manager.Queue(config["num_workers"])
    done = manager.Array('i', [0 for i in range(config["num_workers"])])


    #Random seeding
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True


    #Tracking
    run_name = "PPO_{0}_{1}_{2}_{3}".format(config["env_config"]["env_name"], config["env_config"]["num_agents"],
                                            config["env_config"]["coop_chance"], time.time())
    
    wandb.init(
            project="MetaIPPO",
            sync_tensorboard=True,
            config=config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

    current_path = os.path.dirname(os.path.abspath(__file__))
    runs_path = os.path.join(current_path, "PPO_Runs")
    if not os.path.exists(runs_path):
        os.mkdir(runs_path)
    run_path = os.path.join(runs_path, run_name)

    #Build storage
    training_info = {}
    average_reward = {"agent_{0}".format(a): [] for a in range(config["env_config"]["num_agents"])}
    best_average_reward = {"agent_{0}".format(a): 0.0 for a in range(config["env_config"]["num_agents"])}
    average_success_rate = {"agent_{0}".format(a): [] for a in range(config["env_config"]["num_agents"])}
    best_average_success_rate = {"agent_{0}".format(a): 0.0 for a in range(config["env_config"]["num_agents"])}
    prev_best = 0.0


    #Start the game
    global_step = 0
    
    num_updates = config["total_steps"] // config["batch_size"]
    update = 0

    #Start the workers
    ctx = mp.spawn(rollout, args=([policy_dict, train_queue, done, config]), nprocs=config["num_workers"], join=False)

    print("Initializing workers...")
    time.sleep(10)
    

    while True:
        if all(np.array(done, dtype=bool)):
            # Get the data from the workers
            batch = []
            for i in range(config["num_workers"]):
                batch.append(train_queue.get())
            start = time.time()
            storage, next_obs, next_dones, success_rate, goal_line, goal_line_success = build_storage_from_batch(batch, config)
      
            #Compute the advantages for each policy
            start = time.time()
            for a in range(config["env_config"]["num_agents"]):
                advantages, returns = policy_dict["agent_{0}".format(a)].get_advantages( 
                                                                        storage["agent_{0}".format(a)],
                                                                        next_obs["agent_{0}".format(a)],
                                                                        next_dones["agent_{0}".format(a)])
                
                storage["agent_{0}".format(a)]["advantages"] = advantages
                storage["agent_{0}".format(a)]["returns"] = returns

            #Update the policy parameters
            for a in range(config["env_config"]["num_agents"]):
                loss, pg_loss, value_loss, entropy_loss, explained_variance, clip_fracs =  policy_dict["agent_{0}".format(a)].train(storage["agent_{0}".format(a)])
                print("Agent_{0} loss total: {1}; pg loss: {2}; value loss: {3}; entropy loss: {4}; explained variance: {5}; clip fracs: {6}".format(a, loss, pg_loss, 
                                                                                                                    value_loss, entropy_loss, explained_variance, clip_fracs))
                wandb.log({"agent_{0}_loss".format(a): loss,
                           "agent_{0}_pg_loss".format(a): pg_loss,
                           "agent_{0}_value_loss".format(a): value_loss,
                           "agent_{0}_entropy_loss".format(a): entropy_loss,
                           "agent_{0}_explained_variance".format(a): explained_variance,
                           "agent_{0}_clip_fracs".format(a): clip_fracs})
                
            print("Time to update policy: {0}".format(time.time() - start))

            #TODO: Add all the tracking and printing
            training_info[update] = print_info(storage, next_dones, update, average_reward, best_average_reward,
                                                average_success_rate, best_average_success_rate, success_rate,
                                                goal_line, goal_line_success)
            
            for a in range(config["env_config"]["num_agents"]):
                agent_info = training_info[update]["agent_{0}".format(a)]
                wandb.log({"agent_{0}_average_reward".format(a): agent_info["average_reward"],
                           "agent_{0}_average_success_rate".format(a): agent_info["average_success_rate"],
                            "agent_{0}_rolling_average_reward".format(a): agent_info["rolling_average_reward"],
                            "agent_{0}_rolling_average_success_rate".format(a): agent_info["rolling_average_success_rate"],
                            "agent_{0}_completed".format(a): agent_info["completed"],
                            "agent_{0}_achieved_goal".format(a): agent_info["achieved_goal"],
                            "agent_{0}_achieved_goal_success".format(a): agent_info["achieved_goal_success"],
                            "agent_{0}_successes".format(a): agent_info["successes"],
                            "agent_{0}_rewards".format(a): agent_info["reward"]})

            #Save the models for the agents if the sum of the average rewards is greater than the best average reward
            if sum([best_average_reward["agent_{0}".format(a)] for a in range(config["env_config"]["num_agents"])]) > prev_best:
                prev_best = sum([best_average_reward["agent_{0}".format(a)] for a in range(config["env_config"]["num_agents"])])
                for a in range(config["env_config"]["num_agents"]):
                    save_path = os.path.join(run_path, "models_rew_{0}_ep_{1}".format(prev_best, update, a))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save({"model": policy_dict["agent_{0}".format(a)].agent.state_dict(),
                                "optimizer": policy_dict["agent_{0}".format(a)].optimizer.state_dict()}, 
                                save_path + "/agent_{0}_model.pt".format(a))
                print("Saved models for agents with average reward {0}".format(prev_best)) 
                                                                                                       
                                                                                                       
            

            #Restart the workers
            for e in range(config["num_workers"]):
                done[e] = 0
            
            
            update += 1
            if update >= num_updates:
                print("Training finished")
                ctx.join()
                break
        else:
            continue
        