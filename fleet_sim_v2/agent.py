import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv, GATConv
import numpy as np
import time
from helpfunction import gumbel_softmax, onehot_from_logits


def positive_safe_sigmoid(x):
    return torch.sigmoid(x) + 1e-8


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 device):
        super(GCN, self).__init__()
        self.in_feats = in_feats
        self.n_classes = n_classes
        self.g = (dgl.add_self_loop(g)).to(device)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation).to(device))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation).to(device))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, activation=positive_safe_sigmoid).to(device))

    def forward(self, features):
        h = features.to(torch.float32)
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
        return h


class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    ''' DDPG算法 '''

    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device, avaliable_action):
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = TwoLayerFC(state_dim, action_dim,
                                       hidden_dim).to(device)
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.avaliable_action = torch.tensor(
            [[1.0 if aa != -float('inf') else -float('inf') for i, aa in enumerate(avaliable_action)]]).to(device)

    def take_action(self, state, explore=False):
        # if explore:
        #     print("train state:", state)
        # else:
        #     print("eva state:", state)
        action = self.actor(state)
        ava = self.avaliable_action

        if explore:
            action = gumbel_softmax(action, ava, temperature=1.0)
        else:
            action = action * ava
            action = onehot_from_logits(action)

        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)


class MADDPG:  ## 对每个agent都维护一个DDPG算法
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau, gnn, use_gnn=True):
        self.agents = []
        for i in range(env.road_nums):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device, env.roads[i].neighbor_road))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        self.env = env
        self.gnn = gnn
        self.use_gnn = use_gnn
        if self.use_gnn:
            for cur_agent in self.agents:
                cur_agent.actor_optimizer = torch.optim.Adam([
                    {'params': cur_agent.actor.parameters(), 'lr': actor_lr},
                    {'params': self.gnn.parameters(), 'lr': actor_lr},
                ])

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore):
        # print(states)
        # print(self.env.road_nums)

        states = [
            torch.unsqueeze(states[i], 0).clone().detach().requires_grad_(True).to(self.device)
            for i in range(self.env.road_nums)
        ]

        #####
        # states 形状为 [[],[],[]]的列表，里面的元素为tensor，shape为[1,state size]
        ####

        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def obs2gnn_input(self, obs_, next_obs_):
        """
        将obs经过gnn输入
        :return:
        """
        obs = obs_
        next_obs = next_obs_
        batch_size = obs[0].size(0)  # batch的大小
        raw_state_shape = obs[0].size(1)  # raw state 的状态维度长度

        obs_features = torch.zeros(batch_size, (len(self.agents)), self.gnn.in_feats)  # gnn的输入
        next_obs_features = torch.zeros(batch_size, (len(self.agents)), self.gnn.in_feats)  # gnn的输入

        new_obs = []
        new_next_obs = []
        for i in range(len(self.agents)):
            new_obs.append(torch.zeros(batch_size, raw_state_shape))
            new_next_obs.append(torch.zeros(batch_size, raw_state_shape))

        for batch_index in range(batch_size):
            for agent in range(len(self.agents)):
                obs_features[batch_index, agent] = obs[agent][batch_index]
                next_obs_features[batch_index, agent] = next_obs[agent][batch_index]

        obs_features = obs_features.to(self.device)
        next_obs_features = next_obs_features.to(self.device)

        obs_embedding = torch.zeros(batch_size, (len(self.agents)), self.gnn.n_classes)
        next_obs_embedding = torch.zeros(batch_size, (len(self.agents)), self.gnn.n_classes)
        for index in range(batch_size):
            obs_embedding[index] = self.gnn(obs_features[index])
            next_obs_embedding[index] = self.gnn(next_obs_features[index])

        obs_embedding = obs_embedding.permute(1, 0, 2)
        next_obs_embedding = next_obs_embedding.permute(1, 0, 2)

        for agent in range(len(self.agents)):
            new_obs[agent] = obs_embedding[agent].to(self.device)
            new_next_obs[agent] = next_obs_embedding[agent].to(self.device)

        return new_obs, new_next_obs

    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done = sample
        if self.use_gnn:
            obs_, next_obs_ = self.obs2gnn_input(obs, next_obs)
        else:
            obs_ = obs
            next_obs_ = next_obs
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()
        all_target_act = [
            onehot_from_logits(pi(_next_obs), self.agents[i].avaliable_action, False)
            for i, pi, _next_obs in zip([i for i in range(len(self.agents))], self.target_policies, next_obs_)
        ]
        # print("all:", len(all_target_act))
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(
            -1, 1) + self.gamma * cur_agent.target_critic(
            target_critic_input) * (1 - done[i_agent].view(-1, 1))
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs_[i_agent])
        cur_act_vf_in = gumbel_softmax(cur_actor_out, self.agents[i_agent].avaliable_action)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs_)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs), self.agents[i].avaliable_action, False))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)
