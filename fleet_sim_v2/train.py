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

def print_free_order(city):
    print("free order")
    for road in city.roads:
        print("road index", road.id, " ", len([order for order in road.orders if order.is_accepted == False and order.is_overdue == False]))

def print_free_driver(city):
    print("free driver")
    for road in city.roads:
        print("road index", road.id, " ", len([driver for driver in road.drivers if driver.is_serving == False and driver.serving_order == None]))

def evaluate(g, maddpg, n_episode=100, episode_length=3):
    # 对学习的策略进行评估,此时不会进行探索
    env = City(g)
    env.init_diction()
    env.init_driver_distribution_test()
    env.init_order_distribution_test()
    env.init_travel_time_test()
    returns = np.zeros(env.road_nums)
    for i in range(n_episode):
        obs = env.reset()
        obs = torch.tensor(np.array(obs))
        total = 0
        for t_i in range(episode_length):
            if maddpg.use_gnn:
                gcn_output = maddpg.gnn(obs.to(device))
            else:
                gcn_output = obs.to(device)
            actions = maddpg.take_action(gcn_output, explore=False)
            obs, rew, done = env.step(actions)
            obs = torch.tensor(np.array(obs))
            rew = np.array(rew)
            returns += rew 
    return sum((returns / n_episode).tolist())

num_episodes = 20000  # 250000
episode_length = 3  # 每条序列的最大长度
buffer_size = 40000
hidden_dim = 64
actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.95
tau = 1e-2
batch_size = 1024 * 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
update_interval = 300 # 原来为300  
minimal_size = 4200
use_gnn = True

print("use", device)
print("buffer_size: ", buffer_size)
print("batch_size: ", batch_size)
print("use_gnn: ", use_gnn)
time.sleep(1)

u, v = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]), \
       torch.tensor([1, 2, 3, 4, 1, 3, 1, 2, 4, 1, 3, 5, 3, 4])

# u, v = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6]), \
#        torch.tensor([1, 4, 0, 2, 3, 4, 1, 3, 1, 2, 5, 0, 1, 5, 6, 3, 4, 6, 4, 5])

# u, v = torch.tensor([0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]), \
#        torch.tensor([1, 2, 8, 0, 0, 3, 9, 2, 5, 7, 8, 5, 6, 9, 3, 4, 6, 7, 4, 5, 3, 5, 0, 3, 2, 4])

g = create_graph(u, v)

sim = City(g, use_gnn)

sim.init_driver_distribution_test()
sim.init_order_distribution_test()
sim.init_travel_time_test()
sim.init_diction()

replay_buffer = rl_utils.ReplayBuffer(buffer_size)

state_dims = [6, 6, 6, 6, 6, 6]
action_dims = [4, 4, 4, 4, 4, 4]
critic_input_dim = sum(state_dims) + sum(action_dims)
state_dims = [3, 3, 3, 3, 3, 3]
# state_dims = [5, 5, 5, 5, 5, 5]

arg = {
        "wo_gnn_state_dim": [10, 10, 10, 10, 10, 10],
        "wi_gnn_state_dim": [6, 6, 6, 6, 6, 6],
        "action_dim": [4, 4, 4, 4, 4, 4]
    }

gcn_model = GCN(g, in_feats=6, n_hidden=128, n_classes=3, n_layers=3, activation=torch.nn.functional.relu, device=device)
maddpg = MADDPG(sim, device, actor_lr, critic_lr, hidden_dim, state_dims,
                action_dims, critic_input_dim, gamma, tau, gcn_model, use_gnn=use_gnn)

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

t1 = time.time()
begin_training = False
print_reward = 0
i_episode_list = []
saved_reward_list = []
for i_episode in range(num_episodes):
    state = sim.reset()
    state = torch.tensor(np.array(state)).to(device)
    
    # state 是一个list 3维 每维是对应agent的观测
    # ep_returns = np.zeros(len(env.agents))
    total_reward = 0
    action_li = []
    reward_li = []
    for e_i in range(episode_length):
        if maddpg.use_gnn:
            gcn_output = maddpg.gnn(state.to(device))
            
        else:
            gcn_output = state.to(device)
            
        actions = maddpg.take_action(gcn_output, explore=True)
        # print("type action:", type(actions))
        # print("len: ", len(actions))
        # print("type action[0]: ", type(actions[0]))
        # print("action[0]:", actions[0].shape)
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

            begin_training = True
            sample = [stack_array(x) for x in sample]
            for a_i in range(len(maddpg.agents)):
                maddpg.update(sample, a_i)

            # model2 = maddpg.gnn
            # compare_mode(model1, model2)

            maddpg.update_all_targets()
            t2 = time.time()
            print("time cost:", t2 - t1)
            t1 = time.time()
    print_reward += total_reward
    if begin_training and i_episode % 100 == 0:
        print("i_episode:", i_episode, "training reward:", print_reward / 100.0)
        i_episode_list.append(i_episode)
        saved_reward_list.append(print_reward / 100.0)
        print_reward = 0

 

np.save("i_episode_list_wi_gnn_3.npy", i_episode_list)
np.save("saved_reward_list_wi_gnn_3.npy", saved_reward_list)
# ax1 = plt.subplot(3, 1, 1)
# plt.plot(plt_epi, plt_reward_1)
# plt.xlabel('episode')
# plt.ylabel("reward")
# ax2 = plt.subplot(3, 1, 2)
# plt.plot(plt_epi, plt_reward_2)
# plt.xlabel('episode')
# plt.ylabel("reward")
# ax3 = plt.subplot(3, 1, 3)
# plt.plot(plt_epi, plt_reward_3)
# plt.xlabel('episode')
# plt.ylabel("reward")

# plt.show()
