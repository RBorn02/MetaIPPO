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
        self.lstm = nn.LSTM(self.model_config["lstm_in_size"], self.model_config["lstm_hidden_size"], 
                            num_layers=self.model_config["lstm_layers"])
        
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor_mean = self.layer_init(nn.Linear(self.model_config["lstm_hidden_size"], np.prod(self.action_space_shape)), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.action_space_shape)))

        self.critic = self.layer_init(nn.Linear(self.model_config["lstm_hidden_size"], 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.cnn(x.squeeze().transpose(1, 3))
        hidden = F.relu(self.linear_in(hidden))
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
    
    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    
    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden), lstm_state

    def get_lin_input(self, obs_shape):
        o = self.cnn(torch.zeros(1, *obs_shape).transpose(3, 1))
        return int(np.prod(o.size()))
    