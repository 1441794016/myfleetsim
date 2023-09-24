import dgl
import numpy as np
import random
import torch
from driver import Driver
from order import Order


class Road:
    def __init__(self, road_id, length, neighbor_road, numbers_of_roads):
        self.id = road_id

        self.length = length  # road 长度
        self.drivers = []  # 该road上的所有driver
        # self.controllable_driver = []  # 存放该road上可以接受导航决策车辆
        # self.uncontrollable_driver = []  # 存放该road上不需要进行导航决策的车辆
        self.orders = []  # 该road上出现过的所有order，包括待接收，已接受和已过期的订单
        self.traffic_congestion = 10  # 一个道路拥挤指标
        self.neighbor_road = neighbor_road  # 存放该road 在 n=1 跳内可到达的road
        self.travel_time_to_neighbor_road = []  # 到其他road所要花费的时间
        self.numbers_of_roads = numbers_of_roads  # 总共roads的数量

    def number_of_orders_to_be_served(self):
        """
        计算该road上可以被driver接受的订单的数量（即订单还没有被服务且还未过期）
        :return:
        """
        return len([order for order in self.orders if order.is_accepted is False and order.is_overdue is False])

    def number_of_driver_without_passengers(self):
        """
        计算该road上可以与order匹配的driver的数量（即还没有接受订单的车辆）
        :return:
        """
        return len([driver for driver in self.drivers if driver.is_serving is False])

    def number_of_pari_can_be_matched(self):
        """
        计算该road上可以配对的dirver-order对的数量
        :return:
        """
        return min(self.number_of_orders_to_be_served(), self.number_of_driver_without_passengers())

    def order_driver_match(self):
        """
        order随机分配给该road上的driver
        :return:返回match后的总收入
        """
        assignable_number = self.number_of_pari_can_be_matched()
        random.shuffle(self.drivers)  # 打乱drivers列表
        random.shuffle(self.orders)  # 打乱orders列表
        # source = None
        ###
        road_income = np.zeros(self.numbers_of_roads)
        ###
        number_of_pair_has_been_matched = 0
        for driver_index in range(len(self.drivers)):
            if self.drivers[driver_index].is_serving == False:
                for order_index in range(len(self.orders)):
                    if self.orders[order_index].is_accepted == False and self.orders[order_index].is_overdue == False:
                        self.drivers[driver_index].order_match(self.orders[order_index])
                        self.orders[order_index].driver_match(self.drivers[driver_index])
                        source = self.drivers[driver_index].source  # 该driver上一个时刻经过哪个road调度，source是road编号
                        road_income[source] += self.orders[order_index].price

                        number_of_pair_has_been_matched += 1
                        break
            if number_of_pair_has_been_matched >= assignable_number:
                break

        # 对orders和driver列表重新按id序号进行排序
        self.orders.sort(key=lambda x: x.id)
        self.drivers.sort(key=lambda x: x.id)

        return road_income


class City:
    def __init__(self, g, use_gnn):
        """
        :param g: road graph
        """
        self.time_slot = 0  # 时隙计时
        self.max_time_slot = 1440  # 最大的时隙
        self.road_graph = g
        self.road_nums = g.num_nodes()  # 路网图中节点个数，即道路的数量
        self.roads = []  # 该列表用于存放所有道路，元素是Road类型
        self.all_drivers = []  # 存放所有车
        self.diction = None
        self.use_gnn = use_gnn

        for i in range(self.road_nums):
            road_id = i
            neigh = []
            index = 0
            for node_index in self.road_graph.edges()[0]:
                if node_index.item() == i:
                    neigh.append(self.road_graph.edges()[1][index].item())
                index += 1

            if len(neigh) < 4:
                for j in range(4 - len(neigh)):
                    neigh.append(float('-inf'))

            length = self.road_graph.ndata['length'][i]  # tensor类型
            # frontier = self.road_graph.sample_neighbors(i, -1)  # 查询节点i的所有一阶邻居 有误！ 实际上得到的是是指向该点的点
            # neighbor_road = frontier.edges()[0]  # tensor类型
            self.roads.append(Road(road_id=road_id, length=length, neighbor_road=neigh, numbers_of_roads=self.road_nums))

    def init_driver_distribution(self):
        """
        初始化，把driver分布到每一条road上
        测试：在道路0上初始化4个driver
        :return:
        """
        for i in range(4):
            self.all_drivers.append(Driver(i))

    def init_order_distribution(self):
        pass

    def init_driver_distribution_test(self):
        """
        在road 0 上初始化若干个司机
        :return:
        """
        for i in range(4):
            self.roads[0].drivers.append(Driver(i))

    def init_order_distribution_test(self):
        """
        初始化订单数量
        :return:
        """
        for i in range(12):
            if i < 2:
                self.roads[4].orders.append(Order(i))
            elif i < 4:
                self.roads[5].orders.append(Order(i))
            else:
                self.roads[3].orders.append(Order(i))

    def init_travel_time_test(self):
        # self.roads[0].travel_time_to_neighbor_road = [10, 0, 0, 0, 0, 0]
        # self.roads[1].travel_time_to_neighbor_road = [10, 10, 360, 10, 0, 0]
        # self.roads[2].travel_time_to_neighbor_road = [10, 10, 10, 0, 0, 0]
        # self.roads[3].travel_time_to_neighbor_road = [10, 10, 0, 0, 0, 0]
        # self.roads[4].travel_time_to_neighbor_road = [10, 10, 360, 0, 0, 0]
        # self.roads[5].travel_time_to_neighbor_road = [360, 360, 360, 360, 0, 0]
        # self.roads[6].travel_time_to_neighbor_road = [10, 360, 10, 0, 0, 0]
        # self.roads[7].travel_time_to_neighbor_road = [360, 10, 0, 0, 0, 0]
        
        self.roads[0].travel_time_to_neighbor_road = [10, 0, 0, 0]
        self.roads[1].travel_time_to_neighbor_road = [10, 3000, 10, 0]
        self.roads[2].travel_time_to_neighbor_road = [10, 3000, 0, 0]
        self.roads[3].travel_time_to_neighbor_road = [3000, 3000, 3000, 0]
        self.roads[4].travel_time_to_neighbor_road = [10, 3000, 10, 0]
        self.roads[5].travel_time_to_neighbor_road = [3000, 10, 0, 0]

        # self.roads[0].travel_time_to_neighbor_road = [100, 50, 0, 0]
        # self.roads[1].travel_time_to_neighbor_road = [100, 80, 3000, 80]
        # self.roads[2].travel_time_to_neighbor_road = [70, 3000, 0, 0]
        # self.roads[3].travel_time_to_neighbor_road = [3000, 3000, 3000, 0]
        # self.roads[4].travel_time_to_neighbor_road = [90, 100, 3000, 50]
        # self.roads[5].travel_time_to_neighbor_road = [3000, 80, 60, 0]
        # self.roads[6].travel_time_to_neighbor_road = [60, 60, 0, 0]

        # self.roads[0].travel_time_to_neighbor_road = [10, 10, 10, 0]
        # self.roads[1].travel_time_to_neighbor_road = [10, 0, 0, 10]
        # self.roads[2].travel_time_to_neighbor_road = [10, 60000, 0, 0]
        # self.roads[3].travel_time_to_neighbor_road = [60000, 10, 10, 10]
        # self.roads[4].travel_time_to_neighbor_road = [10, 10, 10, 0]
        # self.roads[5].travel_time_to_neighbor_road = [10, 10, 10, 10]
        # self.roads[6].travel_time_to_neighbor_road = [10, 10, 0, 0]
        # self.roads[7].travel_time_to_neighbor_road = [10, 10, 0, 0]
        # self.roads[8].travel_time_to_neighbor_road = [10, 10, 0, 0]
        # self.roads[9].travel_time_to_neighbor_road = [10, 10, 0, 0]
    def init_diction(self):
        # self.diction = {0: {1: 10}, 1: {0: 10, 2: 10, 5: 360, 6: 10}, 2: {1: 10, 3: 10, 4: 10},
        #                 3: {2: 10, 4: 10}, 4: {2: 10, 3: 10, 5: 360}, 5: {1: 360, 4: 360, 6: 360, 7: 360},
        #                 6: {1: 10, 5: 360, 7: 10}, 7: {5: 360, 6: 10}}
        # self.diction = {0: {1: 10}, 1: {2: 10, 3: 3000, 4: 10}, 2: {3: 3000},
        #                 3: {4: 3000}, 4: {3: 3000, 5: 10}, 5: {3: 3000},
        #                 }
        self.diction = {0: {1: 10}, 1: {2: 10, 3: 3000, 4: 10}, 2: {1: 10, 3: 3000},
                         3: {1: 3000, 2: 3000, 4: 3000}, 4: {1: 10, 3: 3000, 5: 10}, 5: {3: 3000, 4: 10}
                        }
        # self.diction = {0: {1: 100, 4: 50}, 1: {0: 100, 2: 80, 3: 3000, 4: 80}, 2: {1: 70, 3: 3000},
        #                  3: {1: 3000, 2: 3000, 5: 3000}, 4: {0: 90, 1: 100, 5: 3000, 6: 50}, 5: {3: 3000, 4: 80, 6: 60},
        #                  6: {4: 60, 5: 60}
        #                 }
        # self.diction = {0: {1: 10, 2: 10, 8: 10}, 1: {0: 10}, 2: {0: 10, 3: 60000, 9: 10},
        #                 3: {2: 60000, 5: 10, 7: 10, 8: 10, }, 4: {5: 10, 6: 10, 9: 10}, 5: {3: 10, 4: 10, 6: 10, 7: 10},
        #                 6: {4: 10, 5: 10}, 7: {3: 10, 5: 10}, 8: {0: 10, 3: 10}, 9: {2: 10, 4: 10}
        #                 }
    def init_traffic_distribution_test(self):
        """
        初始化道路的交通状况
        :return:
        """
        for i in range(8):
            self.roads[i].traffic_congestion = 30
        self.roads[3].traffic_congestion = 100


    def get_road_observation_test(self):
        observ = []
        if self.use_gnn:
            observation = np.zeros((self.road_nums, 6))  # 使用gnn的话把节点状态设置为6维
            for road_index in range(self.road_nums):
                # 第一维为road上可用车辆的数量
                observation[road_index][0] = float(len([driver for driver in self.roads[road_index].drivers
                                                if driver.is_serving == False and driver.next_road == None]))
                # 第二维为road上order的数量
                observation[road_index][1] = float(len([order for order in self.roads[road_index].orders
                                                if order.is_accepted == False and order.is_overdue == False]))
                # 剩下维度表示到相邻的区域的时间
                inf_count = self.roads[road_index].neighbor_road.count(float('-inf'))
                
                for j in range(4):
                    if j < 4 - inf_count:  
                        observation[road_index][j + 2] = float(self.diction[road_index][self.roads[road_index].neighbor_road[j]])
                        # a = self.roads[self.roads[road_index].neighbor_road[j]].travel_time_to_neighbor_road[j]
                    else:
                        observation[road_index][j + 2] = 0.0
                observ.append(observation[road_index])
        else:
            observation = np.zeros((self.road_nums, 10))  # 不使用gnn的话 把状态设置为10维，相比使用gnn多了临近4个区域的未接订单数
            for road_index in range(self.road_nums):
                # 第一维为road上可用车辆的数量
                observation[road_index][0] = float(len([driver for driver in self.roads[road_index].drivers
                                                if driver.is_serving == False and driver.next_road == None]))
                # 第二维为road上order的数量
                observation[road_index][1] = float(len([order for order in self.roads[road_index].orders
                                                if order.is_accepted == False and order.is_overdue == False]))
                # 剩下维度表示到相邻的区域的时间
                inf_count = self.roads[road_index].neighbor_road.count(float('-inf'))
                
                for j in range(4):
                    if j < 4 - inf_count:  
                        # print("j", j)
                        # print("road_index:", road_index)
    
                        observation[road_index][j + 2] = float(self.diction[road_index][self.roads[road_index].neighbor_road[j]])
                        observation[road_index][j + 4] = float(len([order for order in (self.roads[self.roads[road_index].neighbor_road[j]]).orders if order.is_accepted == False and\
                                                        order.is_overdue == False]))
                        # a = self.roads[self.roads[road_index].neighbor_road[j]].travel_time_to_neighbor_road[j]
                    else:
                        observation[road_index][j + 2] = 0.0
                        observation[road_index][j + 4] = 0.0
                observ.append(observation[road_index])
        return observ

    def assign_order(self):
        """
        Randomly assign call to the drivers at the same road.
        对每一条road都分配订单
        :return:
        """
        total_income = np.zeros(self.road_nums)  # 总收入,np数组
        for road_index in range(self.road_nums):
            income = self.roads[road_index].order_driver_match()
            total_income = [data_i + data_j for data_i, data_j in zip(total_income, income)]
        return total_income

    def update_driver_status(self):
        """
        采取动作后将driver的状态更新为下一个时隙开始时的状态
        暂时假设在时隙开始时已经给出将要去的
        :return:
        """
        for road_index in range(self.road_nums):
            for driver_index in range(len(self.roads[road_index].drivers)):
                if self.roads[road_index].drivers[driver_index].is_serving == False and \
                        self.roads[road_index].drivers[driver_index].next_road != None:  # 对被给出调度决策的车进行状态转移
                    self.roads[road_index].drivers[driver_index].now_location_road = \
                        self.roads[road_index].drivers[driver_index].next_road  # 进入下一个road，更新now_location_road标记
                    self.roads[road_index].drivers[driver_index].next_road = None  # 将要去的road置为None，表示还没有进行决策

    def update_road_status(self):
        """
        对每个road进行状态更新，比如road上的drivers列表
        :return:
        """
        for road_index in range(self.road_nums):
            road_driver_list_temp = self.roads[road_index].drivers
            delete_driver_index = []  # 存放被更新的driver的位置(删除该road上location标志与road不符的driver)

            for driver_index in range(len(self.roads[road_index].drivers)):
                if self.roads[road_index].drivers[driver_index].now_location_road != self.roads[road_index].id:
                    # 如果有driver目前所在的road编号与所在的road不符，则需要更新
                    location_id = self.roads[road_index].drivers[driver_index].now_location_road
                    driver = self.roads[road_index].drivers[driver_index]
                    self.roads[int(location_id)].drivers.append(driver)  # 把driver更新到新的road上
                    delete_driver_index.append(driver_index)

            ######
            # 下面的代码进行删除driver操作
            ######
            for counter, index in enumerate(delete_driver_index):
                index = index - counter
                road_driver_list_temp.pop(index)
            self.roads[road_index].drivers = road_driver_list_temp

    def generate_order(self):
        """
        生成order，将其分配在roads上
        :return:
        """
        pass

    def reset(self):
        """
        reset 环境
        :return:
        """
        self.time_slot = 0
        for road_idx in range(self.road_nums):
            self.roads[road_idx].orders.clear()
            self.roads[road_idx].drivers.clear()

        self.init_driver_distribution_test()
        self.init_order_distribution_test()
        return self.get_road_observation_test()

    def apply_action(self, action):
        """
        采取动作, 这里是对一辆车采取动作；
        由于以道路为agent，agent对其上的所有可调度车辆是顺序给出调度决策，所以这个函数需要调用多次
        :param action: 动作概率 是该city的所有road上所有可控车辆的action概率集，用[[],[]...,[]]表示
                        列表里的元素是numpy数组（或tensor），表示一条road上车辆的action集合
        :return:
        """
        # for driver in road.driver:
        #     if driver.is_controllable:
        #         driver.next_road_id = np.argmax(policy[driver.driver_id])  # 执行动作
        #         driver.is_controllable = False

        # for road in self.roads:
        #     for driver in road.drivers:
        #         if driver.is_controllable is True:

        i = 0
        reward = np.zeros(self.road_nums)
        for road_index in range(self.road_nums):
            j = 0
            number_of_driver_dispatched = 0
            for driver_index in range(len(self.roads[road_index].drivers)):
                driver = self.roads[road_index].drivers[driver_index]
                if driver.is_serving == False and driver.next_road == None:  # 对没接单且没有做出路径决策的车执行动作
                    # print("action:", action[i])
                    # print("roads[road_index].neighbor_road", self.roads[road_index].neighbor_road)
                    next_road = int(np.random.choice(self.roads[road_index].neighbor_road, p=action[i]))
                    next_road_free_order_number = len([order for order in self.roads[next_road].orders if order.is_accepted == False and order.is_overdue == False])
                    # next_road = int(np.random.choice(self.roads[road_index].neighbor_road, p=action[i][j]))
                    self.roads[road_index].drivers[driver_index].next_road = next_road
                    reward[road_index] -= self.diction[road_index][next_road]
                    reward[road_index] += next_road_free_order_number * 300
                    self.roads[road_index].drivers[driver_index].source = road_index
                    j += 1
                    number_of_driver_dispatched += 1
            if number_of_driver_dispatched != 0:
                reward[road_index] /= number_of_driver_dispatched
            else:
                reward[road_index] = 0
            i += 1

        return reward

    def step(self, action):
        """
        执行动作，环境step
        :return:
        """
        reward = self.apply_action(action)
        self.update_driver_status()
        self.update_road_status()
        r = self.assign_order()  # 进行订单分配,返回每条路上被分配的订单价格和
        r = reward + r
        obs_next = self.get_road_observation_test()  # 更新道路状态
        self.time_slot += 1

        done_n = []

        if self.time_slot <= 2:
            for i in range(self.road_nums):
                done_n.append(False)
        else:
            for i in range(self.road_nums):
                done_n.append(True)

        return obs_next, r, done_n


if __name__ == "__main__":

    # u, v = torch.tensor([0, 1, 1, 1, 2, 2, 3, 4, 5, 6, 6, 7]), \
    #        torch.tensor([1, 2, 5, 6, 3, 4, 4, 5, 6, 5, 7, 5])
    u, v = torch.tensor([0, 1, 1, 1, 2, 3, 4, 4, 5]), \
           torch.tensor([1, 2, 3, 4, 3, 4, 3, 5, 3])
    g = dgl.graph((u, v))
    g.ndata['length'] = torch.randn(g.num_nodes())
    for i in range(g.num_nodes()):
        g.ndata['length'][i] = 100
    city = City(g)
    # for i in range(city.road_nums):
    #     print(i)
    #     print(city.roads[i].neighbor_road)
    city.init_driver_distribution_test()
    city.init_order_distribution_test()
    city.init_travel_time_test()
    city.init_diction()

    ac_1 = np.zeros(([3, ]))
    ac_1 = torch.from_numpy(ac_1)
    ac_1[0] = 1
    ac_2 = np.zeros(([3, ]))
    ac_2 = torch.from_numpy(ac_2)
    ac_2[0] = 1
    ac_3 = np.zeros(([3, ]))
    ac_3 = torch.from_numpy(ac_3)
    ac_3[0] = 1
    ac_4 = np.zeros(([3, ]))
    ac_4 = torch.from_numpy(ac_4)
    ac_4[0] = 1
    ac_5 = np.zeros(([3, ]))
    ac_5 = torch.from_numpy(ac_5)
    ac_5[0] = 1
    ac_6 = np.zeros(([3, ]))
    ac_6 = torch.from_numpy(ac_6)
    ac_6[0] = 1
    ac_7 = np.zeros(([3, ]))
    ac_7 = torch.from_numpy(ac_7)
    ac_7[0] = 1
    ac_8 = np.zeros(([3, ]))
    ac_8 = torch.from_numpy(ac_8)
    ac_8[0] = 1

    action = [ac_1, ac_2, ac_3, ac_4, ac_5, ac_6, ac_7, ac_8]
    obs_next, r, done_n = city.step(action)
    print("obs_next:", obs_next)
    print("r:", r)
    print("done:", done_n)

    ac_1 = np.zeros(([3, ]))
    ac_1 = torch.from_numpy(ac_1)
    ac_1[0] = 1
    ac_2 = np.zeros(([3, ]))
    ac_2 = torch.from_numpy(ac_2)
    ac_2[2] = 1
    ac_3 = np.zeros(([3, ]))
    ac_3 = torch.from_numpy(ac_3)
    ac_3[1] = 1
    ac_4 = np.zeros(([3, ]))
    ac_4 = torch.from_numpy(ac_4)
    ac_4[1] = 1
    ac_5 = np.zeros(([3, ]))
    ac_5 = torch.from_numpy(ac_5)
    ac_5[1] = 1
    ac_6 = np.zeros(([3, ]))
    ac_6 = torch.from_numpy(ac_6)
    ac_6[1] = 1
    ac_7 = np.zeros(([3, ]))
    ac_7 = torch.from_numpy(ac_7)
    ac_7[1] = 1
    ac_8 = np.zeros(([3, ]))
    ac_8 = torch.from_numpy(ac_8)
    ac_8[1] = 1

    action = [ac_1, ac_2, ac_3, ac_4, ac_5, ac_6, ac_7, ac_8]
    obs_next, r, done_n = city.step(action)
    print("obs_next:", obs_next)
    print("r:", r)
    print("done:", done_n)

    ac_1 = np.zeros(([3, ]))
    ac_1 = torch.from_numpy(ac_1)
    ac_1[0] = 1
    ac_2 = np.zeros(([3, ]))
    ac_2 = torch.from_numpy(ac_2)
    ac_2[2] = 1
    ac_3 = np.zeros(([3, ]))
    ac_3 = torch.from_numpy(ac_3)
    ac_3[1] = 1
    ac_4 = np.zeros(([3, ]))
    ac_4 = torch.from_numpy(ac_4)
    ac_4[1] = 1
    ac_5 = np.zeros(([3, ]))
    ac_5 = torch.from_numpy(ac_5)
    ac_5[1] = 1
    ac_6 = np.zeros(([3, ]))
    ac_6 = torch.from_numpy(ac_6)
    ac_6[1] = 1
    ac_7 = np.zeros(([3, ]))
    ac_7 = torch.from_numpy(ac_7)
    ac_7[1] = 1
    ac_8 = np.zeros(([3, ]))
    ac_8 = torch.from_numpy(ac_8)
    ac_8[1] = 1

    action = [ac_1, ac_2, ac_3, ac_4, ac_5, ac_6, ac_7, ac_8]
    obs_next, r, done_n = city.step(action)
    print("obs_next:", obs_next)
    print("r:", r)
    print("done:", done_n)

