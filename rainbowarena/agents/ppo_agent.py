import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []  # 动作
        self.states = []  # 状态
        self.logprobs = []  # 动作概率
        self.rewards = []  # 奖励
        self.state_values = []  # 值
        self.is_terminals = []  # 结束
        self.action_masks = []  # 掩码

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.action_masks[:]

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # self.pre_ac = nn.Flatten()

        self.actor = nn.Sequential(
            nn.Linear(state_dim - 156, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.num_actions = action_dim

    def forward(self):
        raise NotImplementedError

    def act(self, state, masks):

        # state = self.pre_ac(state)
        state_part1 = torch.split(state, [330, 156], dim=-1)[0]

        action_probs = self.actor(state_part1)

        dist = CategoricalMasked(logits=action_probs, masks=masks)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action, masks):

        # state = self.pre_ac(state)
        state_part1 = torch.split(state, [330, 156], dim=-1)[0]

        action_probs = self.actor(state_part1)

        dist = CategoricalMasked(logits=action_probs, masks=masks)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

    def eval_act(self, state, masks):

        # state = self.pre_ac(state)
        state_part1 = torch.split(state, [330, 156], dim=-1)[0]

        action_probs = self.actor(state_part1)

        dist = CategoricalMasked(logits=action_probs, masks=masks)

        probs = dist.probs
        action = torch.argmax(probs)

        return action.detach()

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=0.00005, lr_critic=0.00005, gamma=0.99, K_epochs=5, eps_clip=0.1):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_raw = False

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def step(self, state):

        with torch.no_grad():
            obs = state['obs']
            masks = [0] * self.action_dim
            legal_actions = list(state['legal_actions'].keys())
            for legal_action in legal_actions:
                masks[legal_action] = 1
            masks = torch.FloatTensor(masks).to(device)
            obs = torch.FloatTensor(obs).to(device)
            action, action_logprob, state_val = self.policy_old.act(obs, masks)

        self.buffer.states.append(obs)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        self.buffer.action_masks.append(masks)

        return action.item()

    def eval_step(self, state):

        with torch.no_grad():
            obs = state['obs']
            masks = [0] * self.action_dim
            legal_actions = list(state['legal_actions'].keys())
            for legal_action in legal_actions:
                masks[legal_action] = 1
            masks = torch.FloatTensor(masks).to(device)
            obs = torch.FloatTensor(obs).to(device)
            action = self.policy_old.eval_act(obs, masks)

        empty_map = {}

        return action.item(),empty_map

    def update(self):
        # Monte Carlo estimate of returns

        rewards = []

        if self.buffer.is_terminals[-1]:
            discounted_reward = 0
        else:
            discounted_reward = self.policy_old.critic(self.buffer.states[-1]).item()

        for reward in reversed(self.buffer.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        old_action_masks = torch.squeeze(torch.stack(self.buffer.action_masks, dim=0)).detach().to(device)


        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_action_masks)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
