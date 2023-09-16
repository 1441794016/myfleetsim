import torch
import torch.nn.functional as F
import numpy as np
import rl_utils
from agent import MADDPG
import matplotlib.pyplot as plt
import sys
from city import City
import dgl
import time
from agent import GCN


def print_neighbor_road(city):
    for road in city.roads:
        print(road)
        print(road.neighbor_road)


# 获取参数
def compare_mode(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not np.array_equal(p1.detach().numpy(), p2.detach().numpy()):
            print("Parameters are not equal.")
            return False
    print("Parameters are equal.")
    return True
    # 比较两个模型是否参数相同


def create_graph(u, v):
    g = dgl.graph((u, v))
    g.ndata['length'] = torch.randn(g.num_nodes())
    for i in range(g.num_nodes()):
        g.ndata['length'][i] = 100
    return g


num_episodes = 250000
episode_length = 3  # 每条序列的最大长度
buffer_size = 80000
hidden_dim = 64
actor_lr = 1e-2
critic_lr = 1e-2
gamma = 0.95
tau = 1e-2
batch_size = 1024 * 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use", device)
time.sleep(1)
update_interval = 300
minimal_size = 4000

u, v = torch.tensor([0, 1, 1, 1, 2, 3, 4, 4, 5]), \
       torch.tensor([1, 2, 3, 4, 3, 4, 3, 5, 3])
g = create_graph(u, v)

sim = City(g)

sim.init_driver_distribution_test()
sim.init_order_distribution_test()
sim.init_travel_time_test()
sim.init_diction()

replay_buffer = rl_utils.ReplayBuffer(buffer_size)

state_dims = [5, 5, 5, 5, 5, 5]
action_dims = [3, 3, 3, 3, 3, 3]
critic_input_dim = sum(state_dims) + sum(action_dims)
state_dims = [3, 3, 3, 3, 3, 3]
# state_dims = [5, 5, 5, 5, 5, 5]
gcn_model = GCN(g, in_feats=5, n_hidden=64, n_classes=3, n_layers=2, activation=torch.nn.functional.relu, device=device)
maddpg = MADDPG(sim, device, actor_lr, critic_lr, hidden_dim, state_dims,
                action_dims, critic_input_dim, gamma, tau, gcn_model, use_gnn=True)

return_list = []  # 记录每一轮的回报（return）
total_step = 0
plt_reward_1 = []
plt_reward_2 = []
plt_reward_3 = []
plt_epi = []
max_return = float('-inf')

# torch.save(maddpg.gnn, 'gnn.pth')
# model1 = torch.load('gnn.pth')

# gcn_output = gcn_model(state)
max_reward_counter = 0

for i_episode in range(num_episodes):
    state = sim.reset()
    state = torch.tensor(np.array(state)).to(device)

    print("i_episode:", i_episode)
    # state 是一个list 3维 每维是对应agent的观测
    # ep_returns = np.zeros(len(env.agents))
    total_reward = 0
    for e_i in range(episode_length):
        if maddpg.use_gnn:
            gcn_output = maddpg.gnn(state.to(device))
        else:
            state = state.to(device)
        actions = maddpg.take_action(gcn_output, explore=True)
        next_state, reward, done = sim.step(actions)
        
        replay_buffer.add(state, actions, reward, next_state, done)
        state = next_state
        state = torch.tensor(np.array(state))
        total_reward += np.array(reward).sum()
        total_step += 1

        if replay_buffer.size(
        ) >= minimal_size and total_step % update_interval == 0:
            sample = replay_buffer.sample(batch_size)

            def stack_array(x):
                rearranged = [[sub_x[i] for sub_x in x]
                              for i in range(len(x[0]))]
                return [
                    torch.FloatTensor(np.vstack(aa)).to(device)
                    for aa in rearranged
                ]


            sample = [stack_array(x) for x in sample]
            for a_i in range(len(maddpg.agents)):
                maddpg.update(sample, a_i)

            # model2 = maddpg.gnn
            # compare_mode(model1, model2)

            maddpg.update_all_targets()
    if total_reward == 250:
        max_reward_counter += 1
    if i_episode % 100 == 0:
        print("max reward percent: ", max_reward_counter / 100.0)
        max_reward_counter = 0
    

ax1 = plt.subplot(3, 1, 1)
plt.plot(plt_epi, plt_reward_1)
plt.xlabel('episode')
plt.ylabel("reward")
ax2 = plt.subplot(3, 1, 2)
plt.plot(plt_epi, plt_reward_2)
plt.xlabel('episode')
plt.ylabel("reward")
ax3 = plt.subplot(3, 1, 3)
plt.plot(plt_epi, plt_reward_3)
plt.xlabel('episode')
plt.ylabel("reward")

plt.show()
