import dgl
import numpy as np
import torch
from driver import Driver
from order import Order


class Road:
    def __init__(self, road_id, length, neighbor_road):
        self.id = road_id

        self.length = length  # road 长度
        self.drivers = []  # 该road上的所有driver
        self.controllable_driver = []  # 存放该road上可以接受导航决策车辆
        self.uncontrollable_driver = []  # 存放该road上不需要进行导航决策的车辆
        self.orders = []  # 该road上的order
        self.traffic_congestion = 0  # 一个道路拥挤指标
        self.neighbor_road = neighbor_road  # 存放该road 在 n=1 跳内可到达的road


class City:
    def __init__(self, g):
        """
        :param g: road graph
        """
        self.time_slot = 0  # 时隙计时
        self.max_time_slot = 1440  # 最大的时隙
        self.road_graph = g
        self.road_nums = g.num_nodes()  # 路网图中节点个数，即道路的数量
        self.roads = []  # 该列表用于存放所有道路，元素是Road类型
        self.all_drivers = []  # 存放所有车
        for i in range(self.road_nums):
            road_id = i
            length = self.road_graph.ndata['length'][i]  # tensor类型
            frontier = self.road_graph.sample_neighbors(i, -1)  # 查询节点i的所有一阶邻居
            neighbor_road = frontier.edges()[0]  # tensor类型
            self.roads.append(Road(road_id=road_id, length=length, neighbor_road=neighbor_road))

    def init_driver_distribution(self):
        """
        初始化，把driver分布到每一条road上
        测试：在道路0上初始化4个driver
        :return:
        """
        for i in range(4):
            self.all_drivers.append(Driver(i))

    def init_driver_distribution_test(self):
        """
        用于测试 每条路上初始化两个司机
        :return:
        """
        i = 0
        for road in self.roads:
            road.drivers.append(Driver(i))
            i += 1
            road.drivers.append(Driver(i))
            i += 1

    def init_order_distribution(self):
        pass

    def init_order_distribution_test(self):
        i = 0
        for road in self.roads:
            road.orders.append(Order(i))
            i += 1
            road.orders.append(Order(i))
            i += 1
            road.orders.append(Order(i))
            i += 1

    def get_road_observation(self):
        """
        得到每条road的observation
        :return:
        """
        all_road_observation = np.zeros((self.road_nums, 12))
        ########
        # 暂定每个road的observation为12维
        # 通过对地图上最大的邻居road数量计算，最大为5
        # 邻居加上本身所在的road，共6条road
        # 每个road的属性有拥挤程度和待服务的order数量
        # 所以共12维
        ########
        observ = []
        observation = np.zeros((self.road_nums, 12))
        for i in range(self.road_nums):
            inf_count = self.roads[i].neighbor_road.count(float('-inf'))
            for j in range(6):
                if j < 6 - inf_count:
                    if j == 0:
                        observation[i][j] = len([order for order in self.roads[i].orders if order.is_accepted is False])
                        observation[i][j + 6] = self.roads[i].traffic_congestion
                    else:
                        observation[i][j] = len(
                            [order for order in self.roads[self.roads[i].neighbor_road[j - 1]].orders \
                             if order.is_accepted is False])
                        observation[i][j + 6] = self.roads[self.roads[i].neighbor_road[0]].traffic_congestion
                else:
                    observation[i][j] = 0
                    observation[i][j + 6] = 0
            observ.append(observation[i])

        return observ

    def assign_order(self):
        """
        Randomly assign call to the drivers at the same road.
        分配订单
        :return:
        """
        total_income = np.zeros(self.road_nums)
        for i in range(self.road_nums):
            road_income = 0
            assignable_order_number = len([x for x in self.roads[i].orders if x.is_accepted is False])
            assignable_driver_number = len([x for x in self.roads[i].drivers if x.current_serving_order is None])
            assignable_number = min(assignable_order_number, assignable_driver_number)
            for j in range(assignable_number):
                self.roads[i].drivers[j].assign_order(self.roads[i].orders[j])
                road_income += self.roads[i].orders[j].price
            total_income[i] = road_income

        return total_income
        # for road in self.roads:
        #
        #     assignable_order_number = len([x for x in road.orders if x.is_accepted is False])
        #     assignable_driver_number = len([x for x in road.drivers if x.current_serving_order is None])
        #     assignable_number = min(assignable_order_number, len(assignable_driver_number))
        #     for i in range(assignable_number):
        #         driver = road.drivers[i]
        #         driver.assign_order(road.orders[i])

    def assign_order_new(self):
        """
        分配订单
        用于测试
        :return:
        """

        total_income = np.zeros(self.road_nums)
        for i in range(self.road_nums):
            road_income = 0
            assignable_order_number = len([x for x in self.roads[i].orders if x.is_accepted is False])
            assignable_driver_number = len([x for x in self.roads[i].drivers if x.current_serving_order is None])
            assignable_number = min(assignable_order_number, assignable_driver_number)
            for j in range(assignable_number):
                for driver in self.all_drivers:
                    if driver.located_road == j
            for j in range(assignable_number):
                self.roads[i].drivers[j].assign_order(self.roads[i].orders[j])
                road_income += self.roads[i].orders[j].price
            total_income[i] = road_income

        return total_income

    def update_driver_status(self):
        """
        采取动作后将driver的状态更新为下一个时隙开始时的状态
        :return:
        """
        for driver in self.all_drivers:
            if driver.is_controllable is False:  # False表示该辆车已经给过指示了，假设driver将在时隙末到达目的地
                driver.located_road = driver.next_road_index  # 在时隙末driver到达他的目的地
                driver.is_controllable = True  # 重新设置为True

        for road in self.roads:
            for driver in road.uncontrollable_driver:
                driver.located_road = driver.next_road_index

    def update_road_status(self):
        """
        更新road的状态
        :return:
        """
        pass

    def reset(self):
        """
        reset 环境
        :return:
        """
        pass

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

        for road in self.roads:
            i = 0
            for driver in road.controllable_driver:
                j = 0
                next_road = np.random.choice(road.neighbor_road, p=action[i][j])
                self.roads[i].controllable_driver[j].next_road_index = next_road.id
                self.roads[i].uncontrollable_driver.append(self.roads[i].controllable_driver[j])
                j += 1
            i += 1

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
        obs_next = self.get_road_observation()  # 更新道路状态
        self.time_slot += 1

        done_n = []

        if self.time_slot <= 1440:
            for i in range(self.road_nums):
                done_n.append(False)
        else:
            for i in range(self.road_nums):
                done_n.append(False)

        return obs_next, r, done_n
