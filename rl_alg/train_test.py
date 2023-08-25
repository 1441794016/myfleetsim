import torch
import torch.nn.functional as F
import numpy as np
import rl_utils
from agent import MADDPG
import matplotlib.pyplot as plt
import sys
from city import City
import dgl




num_episodes = 150000
episode_length = 25  # 每条序列的最大长度
buffer_size = 100000
hidden_dim = 64
actor_lr = 1e-2
critic_lr = 1e-2
gamma = 0.95
tau = 1e-2
batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
update_interval = 100
minimal_size = 4000

u, v = torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([1, 2, 0, 2, 0, 1])
g = dgl.graph((u, v))
g.ndata['length'] = torch.randn(g.num_nodes())
g.ndata['length'][0] = 100
g.ndata['length'][1] = 100
g.ndata['length'][2] = 100
sim = City(g)
sim.roads[0].traffic_congestion = 30
sim.roads[1].traffic_congestion = 150
sim.roads[2].traffic_congestion = 30
sim.init_driver_distribution_test()
sim.init_order_distribution_test()

replay_buffer = rl_utils.ReplayBuffer(buffer_size)

state_dims = [12, 12, 12]
action_dims = [2, 2, 2]
# print(env.action_space)
# [Discrete(5), Discrete(5), Discrete(5)]
# print(env.action_space[0].n)
# 5
# print(env.observation_space[0].shape[0])
# 18
# for action_space in env.action_space:
#     action_dims.append(action_space.n)
# for state_space in env.observation_space:
#     state_dims.append(state_space.shape[0])
critic_input_dim = sum(state_dims) + sum(action_dims)

# print(critic_input_dim)
# 69
maddpg = MADDPG(sim, device, actor_lr, critic_lr, hidden_dim, state_dims,
                action_dims, critic_input_dim, gamma, tau)

return_list = []  # 记录每一轮的回报（return）
total_step = 0
plt_reward_1 = []
plt_reward_2 = []
plt_reward_3 = []
plt_epi = []
max_return = float('-inf')

for i_episode in range(num_episodes):
    state = sim.reset()
    # state 是一个list 3维 每维是对应agent的观测
    # ep_returns = np.zeros(len(env.agents))
    for e_i in range(episode_length):
        print(e_i)
        actions = maddpg.take_action(state, explore=True)
        next_state, reward, done = sim.step(actions)
        replay_buffer.add(state, actions, reward, next_state, done)
        state = next_state

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
            for a_i in range(3):
                maddpg.update(sample, a_i)
            maddpg.update_all_targets()



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
