import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class LSTMAgent(nn.Module):
    def __init__(self, env, config):
        super(LSTMAgent, self).__init__()
        self.model_config = config["model_config"]
        self.observation_shape = env.vector_env.observation_space.shape
        self.action_space_shape = env.vector_env.action_space.shape
        
        if self.model_config["use_last_action_reward"]:
            lstm_in_size = self.model_config["lstm_in_size"] + np.prod(self.action_space_shape) + 1
        else:
            lstm_in_size = self.model_config["lstm_in_size"]

        if self.model_config["contact"]:
            lstm_in_size += 1
        if self.model_config["time_till_end"]:
            lstm_in_size += 1

        self.cnn = nn.Sequential(
            self.layer_init(nn.Conv2d(self.observation_shape[2], self.model_config["channel_1"], 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(self.model_config["channel_1"], self.model_config["channel_2"], 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(self.model_config["channel_2"], self.model_config["channel_3"], 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(self.model_config["lstm_hidden_size"], self.model_config["critic_hidden_size"])),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.model_config["critic_hidden_size"], self.model_config["critic_hidden_size"])),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.model_config["critic_hidden_size"], 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Linear(self.model_config["lstm_hidden_size"], self.model_config["actor_hidden_size"])),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.model_config["actor_hidden_size"], self.model_config["actor_hidden_size"])),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.model_config["actor_hidden_size"], np.prod(self.action_space_shape)), std=0.01),
        )

        self.linear_in = nn.Linear(self.get_lin_input(self.observation_shape), self.model_config["lstm_in_size"])
        self.lstm = nn.LSTM(lstm_in_size, self.model_config["lstm_hidden_size"], 
                            num_layers=self.model_config["lstm_layers"])
        
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        #self.actor_mean_in = self.layer_init(nn.Linear(self.model_config["lstm_hidden_size"], 128)) #Change std?
        #self.actor_mean_out = self.layer_init(nn.Linear(128, np.prod(self.action_space_shape)), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.action_space_shape))) 

        #self.critic_in = self.layer_init(nn.Linear(self.model_config["lstm_hidden_size"], 128), std=1)
        #self.critic_out = self.layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done, last_action, last_reward, contact, tim_till_end):
        if len(x.shape) == 5:
            hidden = self.cnn(x.squeeze().transpose(1, 3))
        else:
            hidden = self.cnn(x.transpose(1, 3))
        hidden = F.relu(self.linear_in(hidden))
        if self.model_config["use_last_action_reward"]:
            hidden = torch.cat([hidden, last_action, last_reward], dim=1)
        if self.model_config["contact"]:
            hidden = torch.cat([hidden, contact], dim=1)
        if self.model_config["time_till_end"]:
            hidden = torch.cat([hidden, tim_till_end], dim=1)
        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state
    
    def get_value(self, x, lstm_state, done, last_action, last_reward, contact, time_till_end):
        hidden, _ = self.get_states(x, lstm_state, done, last_action, last_reward, contact, time_till_end)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, last_action, last_reward, contact, time_till_end, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done, last_action, last_reward, contact, time_till_end)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        value = self.critic(hidden)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value, lstm_state

    def get_lin_input(self, obs_shape):
        o = self.cnn(torch.zeros(1, *obs_shape).transpose(3, 1))
        return int(np.prod(o.size()))
    
    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    

class CommsLSTMAgent(nn.Module):
    def __init__(self, env, config):
        super(CommsLSTMAgent, self).__init__()
        self.model_config = config["model_config"]
        self.observation_shape = env.observation_space.shape
        self.movement_shape = env.vector_env.action_space["actuators_action_space"].shape
        self.message_shape = env.vector_env.action_space["message_action_space"].shape
        self.message_space = list(env.vector_env.action_space["message_action_space"].nvec)
        self.action_space_shape = (int(np.prod(self.movement_shape) + np.prod(self.message_shape)),)
        
        
        
        if self.model_config["use_last_action_reward"] and self.model_config["one_hot_message"] is False:
            lstm_in_size = self.model_config["lstm_in_size"] + np.prod(self.action_space_shape) + np.prod(self.message_shape) + 1
        elif self.model_config["use_last_action_reward"] and self.model_config["one_hot_message"]:
            lstm_in_size = self.model_config["lstm_in_size"] +  np.prod(self.action_space_shape) + (env.base_env.vocab_size + 1) * env.base_env.message_len  + 1
        elif self.model_config["use_last_action_reward"] is False and self.model_config["one_hot_message"]:
            lstm_in_size = self.model_config["lstm_in_size"] + (env.base_env.vocab_size + 1) * env.base_env.message_len
        else:
            lstm_in_size = self.model_config["lstm_in_size"] + np.prod(self.message_shape)
        
        if self.model_config["contact"]:
            lstm_in_size += 1
        
        if self.model_config["time_till_end"]:
            lstm_in_size += 1

        self.cnn = nn.Sequential(
            self.layer_init(nn.Conv2d(self.observation_shape[2], self.model_config["channel_1"], 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(self.model_config["channel_1"], self.model_config["channel_2"], 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(self.model_config["channel_2"], self.model_config["channel_3"], 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.linear_in = nn.Linear(self.get_lin_input(self.observation_shape), self.model_config["lstm_in_size"])
        self.lstm = nn.LSTM(lstm_in_size, self.model_config["lstm_hidden_size"], 
                            num_layers=self.model_config["lstm_layers"])
        
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(self.model_config["lstm_hidden_size"], self.model_config["critic_hidden_size"])),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.model_config["critic_hidden_size"], self.model_config["critic_hidden_size"])),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.model_config["critic_hidden_size"], 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Linear(self.model_config["lstm_hidden_size"], self.model_config["actor_hidden_size"])),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.model_config["actor_hidden_size"], self.model_config["actor_hidden_size"])),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.model_config["actor_hidden_size"], np.prod(self.movement_shape) + sum(self.message_space)), std=0.01),
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.movement_shape))) 


    def get_states(self, x, lstm_state, done, message, last_action, last_reward, contact, time_till_end):
        if len(x.shape) == 5:
            hidden = self.cnn(x.squeeze().transpose(1, 3))
        else:
            hidden = self.cnn(x.transpose(1, 3))
        hidden = F.relu(self.linear_in(hidden))

        #Handle messages
        if self.model_config["one_hot_message"]:
            message = F.one_hot(message.long(), num_classes=self.message_space[0]).float()
            message = message.reshape(message.shape[0], -1)
        
        if self.model_config["use_last_action_reward"]:
            hidden = torch.cat([hidden, last_action, last_reward], dim=1)

        hidden = torch.cat([hidden, message], dim=1)

        if self.model_config["contact"]:
            hidden = torch.cat([hidden, contact], dim=1)
        
        if self.model_config["time_till_end"]:
            hidden = torch.cat([hidden, time_till_end], dim=1)
        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state
    
    def get_value(self, x, lstm_state, done, message, last_action, last_reward, contact, time_till_end):
        hidden, _ = self.get_states(x, lstm_state, done, message, last_action, last_reward, contact, time_till_end)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, message, last_action, last_reward, contact, time_till_end ,action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done, message, last_action, last_reward, contact, time_till_end)
        actions = self.actor_mean(hidden)

        # Sample movement actions
        action_mean = actions[:, :self.movement_shape[0]]
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        # Sample message actions
        logits = actions[:, self.movement_shape[0]:]
        split_logits = torch.split(logits, self.message_space, dim=1)
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]

        if action is None:
            movement_action = probs.sample()
            message_action = torch.stack([categorical.sample() for categorical in multi_categoricals], dim=1)
        else:
            movement_action = action[:, :self.movement_shape[0]]
            message_action = action[:, self.movement_shape[0]:]

        movement_probs = probs.log_prob(movement_action).sum(1)
        movement_entropy = probs.entropy().sum(1)

        #message_logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(message_action, multi_categoricals)])
        message_probs =[] #Get logprob for each message action, not sure if this is correctly implemented
        for a in range(message_action.shape[1]):
            message_probs.append(multi_categoricals[a].log_prob(message_action[:,a]))
        message_logprob = torch.stack(message_probs, dim=1)
            

        message_entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        probs = torch.add(movement_probs, message_logprob.sum(1))
        entropy = torch.add(movement_entropy, message_entropy.sum(0))

        value = self.critic(hidden)
        actions = torch.cat([movement_action, message_action], dim=1)
        return actions, probs, entropy, value, lstm_state

    def get_lin_input(self, obs_shape):
        o = self.cnn(torch.zeros(1, *obs_shape).transpose(3, 1))
        return int(np.prod(o.size()))
    
    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    