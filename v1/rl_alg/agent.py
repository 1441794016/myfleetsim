import torch
import torch.nn.functional as F
import torch.nn as nn
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
                 activation):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, activation=positive_safe_sigmoid))

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


class GraphEmbeddingActor(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, hidden_dim, num_out, activation):
        super(GraphEmbeddingActor, self).__init__()
        self.g = g
        self.graph_network = nn.ModuleList()
        self.graph_network.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.graph_network.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.graph_network.append(GraphConv(n_hidden, n_classes, activation=positive_safe_sigmoid))

        self.fc1 = torch.nn.Linear(n_classes, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.graph_network):
            h = layer(self.g, h)

        return h


class DDPG:
    ''' DDPG算法 '''

    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device):
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = TwoLayerFC(state_dim, action_dim,
                                       hidden_dim).to(device)
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1,
                                        hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

    def take_action(self, state, explore=False):
        # if explore:
        #     print("train state:", state)
        # else:
        #     print("eva state:", state)
        action = self.actor(state)

        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)

        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)


class MADDPG:  ## 对每个agent都维护一个DDPG算法
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau, gnn):
        self.agents = []
        for i in range(env.road_nums):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        self.env = env
        # self.gnn = GCN(g, in_feats, n_hidden, n_classes, n_layers, activation)
        self.gnn = gnn
        for cur_agent in self.agents:
            cur_agent.actor_optimizer = torch.optim.Adam([
                    {'params': cur_agent.actor.parameters(), 'lr': actor_lr, },
                    {'params': self.gnn.parameters()},
            ])

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def gnn_compute(self, features):
        return self.gnn(features)

    def take_action(self, states, explore):
        # print(states)
        # print(self.env.road_nums)
        states = [
            torch.unsqueeze(states[i], 0)
            # torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(self.env.road_nums)
        ]

        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done = sample

        after_next_obs = []
        for i in range(len(self.agents)):
            after_next_obs.append(torch.ones((next_obs[0].size(0), 2)))
        # after_next_obs = torch.ones((next_obs.size(0), 2))  # 原先的next_obs是未经过gnn前的维度，现在将其输入gnn中
        features = torch.ones((3, 12))
        for i in range(next_obs[0].size(0)):
            for node in range(len(self.agents)):

                features[node] = next_obs[node][i]

            embedding = self.gnn(features)

            for node_ in range(len(self.agents)):
                after_next_obs[node_][i] = embedding[node_]
        # print("len after_next_obs:", len(after_next_obs))
        # print("after_next_obs shape[0]", after_next_obs[0].shape)
        # print("after_next_obs shape[1]", after_next_obs[1].shape)
        # print("after_next_obs shape[2]", after_next_obs[2].shape)
        after_obs = []
        for i in range(len(self.agents)):
            after_obs.append(torch.ones((obs[0].size(0), 2)))
        # after_next_obs = torch.ones((next_obs.size(0), 2))  # 原先的next_obs是未经过gnn前的维度，现在将其输入gnn中
        features = torch.ones((3, 12))
        for i in range(obs[0].size(0)):
            for node in range(len(self.agents)):
                features[node] = obs[node][i]

            embedding = self.gnn(features)

            for node_ in range(len(self.agents)):
                after_obs[node_][i] = embedding[node_]



        # print("obs type:", type(obs))
        # print("obs :", obs[0].shape)
        # print("act type:", type(act))
        # print("act :", act[0].shape)
        # print("rew type:", type(rew))
        # print("rew :", rew[0].shape)
        # print("next_obs type:", type(next_obs))
        # print("next_obs :", next_obs[0].shape)
        # print("done:", type(done))
        # obs, act, rew, next_obs, done 都是长度为3的列表，表示3个agent，列表里面的每个元素都是tensor
        # print("next_obs.shape:", len(next_obs))  # 长度为3的列表
        # print("next_obs[0].shape:", next_obs[0].shape)  # torch.Size([1024, 18])
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()
        all_target_act = [
            onehot_from_logits(pi(_next_obs))
            for pi, _next_obs in zip(self.target_policies, after_next_obs)
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
        cur_actor_out = cur_agent.actor(after_obs[i_agent])
        cur_act_vf_in = gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, after_obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs)))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)