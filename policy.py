import torch
import torch.nn as nn
import numpy as np


class LSTM_PPO_Policy():
    def __init__(self, config, agent, optimizer):
        super(LSTM_PPO_Policy, self).__init__()

        self.config = config
        self.agent = agent
        self.optimizer = optimizer
        self.observation_shape = self.agent.observation_shape
        self.single_action_shape = self.agent.action_space_shape

    def get_value(self, obs, lstm_state, dones, last_action, last_reward):
        """Returns the value of the observation"""
        return self.agent.get_value(obs, lstm_state, dones, last_action, last_reward)
    
    def get_action_and_value(self, obs, lstm_state, done, last_action, last_reward, action=None):
        """Returns the action and value of the observation"""
        return self.agent.get_action_and_value(obs, lstm_state, done, last_action, last_reward, action)
    
    def get_advantages(self, storage, next_obs, next_done):
        """Returns the advantage for the Policy"""
        device = self.config["device"]
        rewards = storage["rewards"].to(device)
        dones = storage["dones"].to(device)
        values = storage["values"].to(device)
        next_lstm_state = storage["next_lstm_state"]
        num_steps = self.config["rollout_steps"] // (self.config["env_config"]["num_envs"] * self.config["num_workers"])

        next_obs = next_obs.to(device)
        next_done = next_done.to(device)
        
        actions = storage["actions"]
        prev_actions = actions[-1]
        prev_rewards = rewards[-1].unsqueeze(dim=1)
    
        with torch.no_grad():
            # TODO: Adapt this for the case with less then 100 percent coop. Doesn't work for other cases!!!!
            # Should probably work to collect an extra step during training and then using this extra step for
            # next value and next done
            next_value = self.agent.get_value(
                next_obs,
                next_lstm_state,
                next_done,
                prev_actions,
                prev_rewards,
            ).reshape(1, -1)
            #next_value = next_step_storage["values"].reshape(1, -1)
            #next_done = next_step_storage["dones"].reshape(1, -1)
            if self.config["gae"]:
                advantages = torch.zeros_like(rewards).to(self.config["device"])
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1] 
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.config["gamma"] * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.config["gamma"] * self.config["gae_lambda"] * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(self.config["device"])
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + self.config["gamma"] * nextnonterminal * next_return
                advantages = returns - values

        return advantages, returns

    def train(self, storage):
        """Takes in the stored rollout and trains the policy
           Based on https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/"""
        
        loss_ls = []
        pg_loss_ls = []
        v_loss_ls = []
        entropy_ls = []
        predicted_values_mean = []
        returns_mean = []
        old_values_mean = []
        advantages_mean = []
        
        device = self.config["device"]

        initial_lstm_state = storage["initial_lstm_state"]
        obs = storage["obs"]
        actions = storage["actions"]
        prev_actions = storage["last_actions"]
        rewards = storage["rewards"]
        prev_rewards = storage["last_rewards"]
        logprobs = storage["logprobs"]
        dones = storage["dones"]
        values = storage["values"]
        advantages = storage["advantages"]
        returns = storage["returns"]
        num_steps = self.config["rollout_steps"] // (self.config["env_config"]["num_envs"] * self.config["num_workers"])

        
        b_obs = obs.reshape((-1,) + self.observation_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.single_action_shape)
        b_prev_actions = prev_actions.reshape((-1,) + self.single_action_shape)
        b_prev_rewards = prev_rewards.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_dones = dones.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        print(self.agent.actor_logstd)

       
        assert self.config["env_config"]["num_envs"] * self.config["num_workers"] % self.config["num_minibatches"] == 0
        envsperbatch = (self.config["env_config"]["num_envs"] * self.config["num_workers"]) // self.config["num_minibatches"]
        envinds = np.arange(self.config["env_config"]["num_envs"] * self.config["num_workers"])
        flatinds = np.arange(self.config["batch_size"]).reshape(num_steps, self.config["env_config"]["num_envs"] * self.config["num_workers"])
        clipfracs = []
        for epoch in range(self.config["update_epochs"]):
            np.random.shuffle(envinds)
            for start in range(0, self.config["env_config"]["num_envs"] * self.config["num_workers"], envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = self.agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_prev_actions[mb_inds],
                    b_prev_rewards[mb_inds].unsqueeze(dim=1),
                     b_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.config["clip_coef"]).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.config["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config["clip_coef"], 1 + self.config["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.config["clip_vloss"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.config["clip_coef"],
                        self.config["clip_coef"],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                predicted_values_mean.append(newvalue.mean().item())
                returns_mean.append(b_returns[mb_inds].mean().item())
                old_values_mean.append(b_values[mb_inds].mean().item())
                advantages_mean.append(b_advantages[mb_inds].mean().item())
                #print(np.mean(predicted_values_mean.item()))
                #print(np.mean(returns_mean))
                #print(np.mean(old_values_mean))

                entropy_loss = entropy.mean()
                loss = pg_loss - self.config["ent_coef"] * entropy_loss + v_loss * self.config["vf_coef"]

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config["max_grad_norm"])
                self.optimizer.step()
                
                loss_ls.append(loss.item())
                pg_loss_ls.append(pg_loss.item())
                v_loss_ls.append(v_loss.item())
                entropy_ls.append(entropy_loss.item())

                if epoch == 0 and start == 0:
                    print("New values mean: {0}; Old values mean: {1}".format(newvalue.mean().item(), b_values[mb_inds].mean().item()))
                    if all(newvalue == b_values[mb_inds]):
                        print("New values and old values are the same")
            
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        print("Predicted values mean: {0}; Returns mean: {1}; Advantages mean: {2}; Values mean: {3}".format(np.mean(predicted_values_mean), 
                                                                    np.mean(returns_mean), np.mean(advantages_mean), np.mean(old_values_mean)))
        print("Actions mean: {0}; Actions std: {1}".format(b_actions.mean(), b_actions.std()))
        print("Action logprob mean: {0}; Action logprob std: {1}".format(b_logprobs.mean(), logprobs.std()))

    
        #Return the mean of the losses
        return np.mean(loss_ls), np.mean(pg_loss_ls), np.mean(v_loss_ls), np.mean(entropy_ls), explained_var, np.mean(clipfracs)