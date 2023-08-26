import torch
import torch.nn.functional as F
import numpy as np
import rl_utils
from agent import MADDPG
import matplotlib.pyplot as plt
import sys
from city import City
import dgl
sys.path.append(".")     ## 指定到上一级目录
from utils.make_env import *

def evaluate(env_id, maddpg, n_episode=10, episode_length=25):
    # 对学习的策略进行评估,此时不会进行探索
    env = make_env(env_id, discrete_action=True)
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        obs = env.reset()
        for t_i in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)
            obs, rew, done, info = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
    return returns.tolist()

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



env_id = "simple_spread"
env = make_env("simple_spread", discrete_action=True)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)

state_dims = []
action_dims = []
# print(env.action_space)
# [Discrete(5), Discrete(5), Discrete(5)]
# print(env.action_space[0].n)
# 5
# print(env.observation_space[0].shape[0])
# 18
for action_space in env.action_space:
    action_dims.append(action_space.n)
for state_space in env.observation_space:
    state_dims.append(state_space.shape[0])
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
            for a_i in range(len(env.agents)):
                maddpg.update(sample, a_i)
            maddpg.update_all_targets()
    if (i_episode + 1) % 100 == 0:
        ep_returns = evaluate(env_id, maddpg, n_episode=100)
        return_list.append(ep_returns)
        print(f"Episode: {i_episode+1}, {ep_returns}")
        if sum(ep_returns) >= max_return:
            max_return = sum(ep_returns)
            print(f"save {i_episode + 1 }")
            torch.save(obj=maddpg.agents[0].actor.state_dict(), f=f"MADDPG/model/ac_net1_best.pth")
            torch.save(obj=maddpg.agents[1].actor.state_dict(), f=f"MADDPG/model/ac_net2_best.pth")
            torch.save(obj=maddpg.agents[2].actor.state_dict(), f=f"MADDPG/model/ac_net3_best.pth")
            # torch.save(obj=maddpg.agents[0].critic.state_dict(), f=f"MADDPG/model/c_net1_best.pth")
            # torch.save(obj=maddpg.agents[1].critic.state_dict(), f=f"MADDPG/model/c_net2_best.pth")
            # torch.save(obj=maddpg.agents[2].critic.state_dict(), f=f"MADDPG/model/c_net3_best.pth")
        plt_reward_1.append(ep_returns[0])
        plt_reward_2.append(ep_returns[1])
        plt_reward_3.append(ep_returns[2])
        plt_epi.append(i_episode+1)
        np.save("MADDPG/model/plt_reward_1.npy",plt_reward_1)
        np.save("MADDPG/model/plt_reward_2.npy",plt_reward_2)
        np.save("MADDPG/model/plt_reward_3.npy",plt_reward_3)
        np.save("MADDPG/model/plt_epi.npy",plt_epi)


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
