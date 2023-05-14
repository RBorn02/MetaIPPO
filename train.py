import random
import torch
import numpy as np
from distutils.util import strtobool

from argparse import ArgumentParser

from policy import LSTM_PPO_Policy
from agent import LSTMAgent
from Envs.environment_handler import EnvironmentHandler
from Utils.train_utils import build_config, build_storage, handle_dones, print_info, get_init_tensors, truncate_storage, reset_storage

parser = ArgumentParser()

parser.add_argument("--config", default=None,
                    help="Optional path to the config yaml")
parser.add_argument("--env_name", type=str, default="MultiAgentLandmarks",
                    help="Name of the environment to use")
parser.add_argument("--num_agents", type=int, default=2,
                    help="Number of agents in the environment")
parser.add_argument("--num_landmarks", type=int, default=3,
                    help="Number of landmarks in the environment")
parser.add_argument("--num_envs", type=int, default=16,
                    help="Number of environments to vectorize")
parser.add_argument("--time_limit", type=int, default=250,
                    help="Number of max steps per episode")
parser.add_argument("--coop_chance", type=float, default=0.5,
                    help="Chance of cooperative goal")
parser.add_argument("--total_steps", type=int, default=2.5*10e7,
                    help="Number of steps to train for")
parser.add_argument("--rollout_steps", type=int, default=16384,
                    help="Number of steps per rollout")
parser.add_argument("--batch_size", type=int, default=4096,
                    help="Size of the minibatch")
parser.add_argument("--seed", type=int, default=1,
                    help="Random seed")
parser.add_argument("--device", type=str, default="cpu",
                    help="Device to use for training")

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




args = parser.parse_args()
args.batch_size = int(args.num_envs * args.rollout_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)

config = build_config(args)

#Build the environemnt
env = EnvironmentHandler(config)

#Build the agents with their corresponding optimizers
agent_dict = {"agent_{0}".format(a): LSTMAgent(env, config) for a in range(config["env_config"]["num_agents"])}
optimizer_dict = {"agent_{0}".format(a): {"optimizer": torch.optim.Adam(agent_dict["agent_{0}".format(a)].parameters(),
                                                                        config["lr"], eps=1e-5)} 
                  for a in range(config["env_config"]["num_agents"])}

#Build the policies
policy_dict = {"agent_{0}".format(a): LSTM_PPO_Policy(config, agent_dict["agent_{0}".format(a)], optimizer_dict["agent_{0}".format(a)]) 
               for a in range(config["env_config"]["num_agents"])}

#Train the agents
if __name__ == "__main__":
    #Random seeding
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True

    #Build storage
    storage = build_storage(config, env)
    training_info = {}

    #Start the game
    global_step = 0
    next_obs, _ = env.reset([i for i in range(config["env_config"]["num_envs"])])
    next_dones = {"agent_{0}".format(a): torch.zeros(config["env_config"]["num_envs"]) for a in range(config["env_config"]["num_agents"])}
    #for a in range(config["env_name"]["num_agents)"]):
    #    storage["agent_{0}".format(a)]["next_obs"] = next_obs["agent_{0}".format(a)]
    
    num_updates = config["total_steps"] // config["batch_size"]

    for update in range(1, int(num_updates + 1)):
        #Get the intial LSTM states for each update. LSTM state for step 1 initialized by the storage builder
        for a in range(config["env_config"]["num_agents"]):
            storage["agent_{0}".format(a)]["initial_lstm_state"] = (storage["agent_{0}".format(a)]["next_lstm_state"][0].clone(),
                                                                    storage["agent_{0}".format(a)]["next_lstm_state"][1].clone())
        
        # Annealing the rate if instructed to do so.
        if config["anneal_lr"]:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config["lr"]
            for a in range(config["env_config"]["num_agents"]):
                optimizer_dict["agent_{0}".format(a)]["optimizer"].param_groups[0]["lr"] = lrnow
        
        #Collect trajectories for the agents. Keep going untill all agents have collected enough steps
        rollout_step = 0
        steps_per_env = torch.Tensor([config["rollout_steps"] // config["env_config"]["num_envs"]]).repeat(config["env_config"]["num_envs"])
        while True:
            global_step += 1 * config["env_config"]["num_envs"]
            for a in range(config["env_config"]["num_agents"]):
                next_agent_obs = next_obs["agent_{0}".format(a)]
                next_agent_dones = next_dones["agent_{0}".format(a)]
                next_agent_lstm_state = storage["agent_{0}".format(a)]["next_lstm_state"]
                
                torch.cat((storage["agent_{0}".format(a)]["obs"], next_agent_obs), dim=0)
                torch.cat((storage["agent_{0}".format(a)]["dones"], next_agent_dones), dim=0)
            
                #Get the actions from the policy
                with torch.no_grad():
                    #Handle the case where one agent is done while the oter is still going. We only want to send 
                    #env obs to policy if it is not done in this env instance. We collect data until all environments
                    #have num (rollout_steps // num_envs) data points. Can be inefficient because some collected data will be discarded.
                    #However, this makes updating the policy much simpler.
                    
                    dones_idx = next_agent_dones.nonzero().squeeze()
                    storage["agent_{0}".format(a)]["is_training"].cat(torch.ones_like(next_agent_dones) - next_agent_dones)
                    storage["agent_{0}".format(a)]["collected_env_steps"] + (torch.ones_like(next_agent_dones) - next_agent_dones).squeeze()

                    action, log_prob, value, next_lstm_state = get_init_tensors(env, config)
                    act_raw, log_prob_raw, _, val_raw, next_lstm_state_raw = policy_dict["agent_{0}".format(a)].get_action_and_value(
                                                                                                                next_agent_obs[dones_idx],
                                                                                                                 next_agent_lstm_state[:,dones_idx,:],
                                                                                                                 next_agent_dones[dones_idx])
                    action[dones_idx] = act_raw
                    log_prob[dones_idx] = log_prob_raw
                    value[dones_idx] = val_raw
                    next_lstm_state[0][:,dones_idx,:] = next_lstm_state_raw[0]

                    
                    actions = {"agent_{0}".format(a): action}

                    storage["agent_{0}".format(a)]["values"].cat(value, dim=0)
                
                storage["agent_{0}".format(a)]["actions"].cat(action, dim=0)
                storage["agent_{0}".format(a)]["log_probs"].cat(log_prob, dim=0)

                storage["agent_{0}".format(a)]["next_lstm_state"][0] = next_lstm_state[0]
                storage["agent_{0}".format(a)]["next_lstm_state"][1] = next_lstm_state[1]

            #Take a step in the environment
            actions = torch.cat([actions["agent_{0}".format(a)] for a in range(config["env_config"]["num_agents"])], dim=1)
            next_obs, rewards, dones, _, infos = env.step(actions)
            
            #Handle the dones and convert the bools to binary tensors
            next_dones = handle_dones(dones)

            #Store the rewards
            for a in range(config["env_config"]["num_agents"]):
                storage["agent_{0}".format(a)]["rewards"].cat(rewards["agent_{0}".format(a)], dim=0)
            
            
            for e in range(config["env_config"]["num_envs"]):
                if dones["__all__"][e]:
                    next_obs[e], _ = env.reset(e)
                    next_dones[e] = {"agent_{0}".format(a): 0.0 for a in range(config["env_config"]["num_agents"])}

                

            #If we have collected enough data for each env for each agent, break the loop
            rollout_step += 1
            condition_ls = []
            for a in range(config["env_config"]["num_agents"]):
                condition = storage["agent_{0}".format(a)]["collected_env_steps"] >= steps_per_env
                condition_ls.append(condition)
            
            if all(condition_ls):
                break

        #Get correct training data from storage and truncate it
        storage = truncate_storage(storage, rollout_step, config)  
            
            
        
        #Compute the advantages for each policy
        for a in range(config["env_config"]["num_agents"]):
            storage["agent_{0}".format(a)]["advantages"] = policy_dict["agent_{0}".format(a)].compute_advantages( 
                                                                                storage["agent_{0}".format(a)],
                                                                                next_obs["agent_{0}".format(a)],
                                                                                next_dones["agent_{0}".format(a)],
            )
        
        #Update the policy parameters
        for a in range(config["env_config"]["num_agents"]):
            policy_dict["agent_{0}".format(a)].train(storage["agent_{0}".format(a)])
        
        #Reset the storage for the next epoch
        storage = reset_storage(storage, config, env)

        #TODO: Add all the tracking and printing
        training_info[update] = print_info(storage, update) 
        
        if global_step >= config["total_steps"]:
            print("Training finished")
            break




