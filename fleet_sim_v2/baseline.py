import numpy as np

class random_agent:
    def __init__(self, ava_action, action_dim):
        self.avaliable_action = ava_action.copy()
        self.prob = []
        self.inf_counter = 0
        self.action_dim = action_dim
        for data in self.avaliable_action:
            if data == float('-inf'):
                self.inf_counter += 1
        

        for i in range(len(self.avaliable_action)):
            if self.avaliable_action[i] != float('-inf'):
                self.avaliable_action[i] = i

        for i in range(len(self.avaliable_action)):
            if i < len(self.avaliable_action) - self.inf_counter:
                self.prob.append(1.0 / float(len(self.avaliable_action) - self.inf_counter))
            else:
                self.prob.append(0)

class random_multi_agent:
    def __init__(self, env):
        self.env = env
        self.agents = []
        self.action_dim = 4
        for i in range(self.env.road_nums):
            self.agents.append(random_agent(env.roads[i].neighbor_road, action_dim=4))
        
    def act(self):
        actions = []
        for i in range(len(self.agents)):

            action = int(np.random.choice(self.agents[i].avaliable_action, p = self.agents[i].prob))
            actions.append(np.eye(self.action_dim, dtype=np.float32)[action])
        return actions


if __name__=="__main__":
    import torch
    from city import City
    import time
    import dgl
    def create_graph(u, v):
        g = dgl.graph((u, v))
        g.ndata['length'] = torch.randn(g.num_nodes())
        for i in range(g.num_nodes()):
            g.ndata['length'][i] = 100
        return g

    u, v = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]), \
           torch.tensor([1, 2, 3, 4, 1, 3, 1, 2, 4, 1, 3, 5, 3, 4])

    use_gnn = False
    g = create_graph(u, v)

    sim = City(g, use_gnn)
    sim.init_driver_distribution_test()
    sim.init_order_distribution_test()
    sim.init_travel_time_test()
    sim.init_diction()
    

    random_agents = random_multi_agent(sim)

    for agent in random_agents.agents:
        print(agent.avaliable_action)
        print(agent.prob)

    num_episodes = 3000
    episode_length = 3 

    re_list = []
    epi_list = []

    for i_episode in range(num_episodes):
        total_reward = 0
        sim.reset()
        for e_i in range(episode_length):            
            actions = random_agents.act()
            next_state, reward, done = sim.step(actions)
            
            total_reward += np.array(reward).sum()

        print("total reward:", total_reward)    
        re_list.append(total_reward)
        epi_list.append(i_episode)

        
    np.save("i_episode_list_rand.npy", epi_list)
    print("ok")
    np.save("reward_list_rand.npy", re_list)