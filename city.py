import dgl
import numpy as np
import torch
from driver import Driver
from order import Order

class Road:
    def __init__(self, road_id, length, neighbor_road):
        self.id = road_id

        self.length = length  # road 长度
        self.driver = []  # 该road上的driver
        self.orders = []  # 该road上的order
        # self.controllable_driver = []  # 存放该road上可以接受导航决策车辆
        # self.uncontrollable_driver = []  # 存放该road上不需要进行导航决策的车辆
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
        for i in range(self.road_nums):
            road_id = i
            length = self.road_graph.ndata['length'][i]  # tensor类型
            frontier = self.road_graph.sample_neighbors(i, -1)  # 查询节点i的所有一阶邻居
            neighbor_road = frontier.edges()[0]  # tensor类型
            self.roads.append(Road(road_id=road_id, length=length, neighbor_road=neighbor_road))
        pass

    def init_driver_distribution(self):
        """
        初始化，把driver分布到每一条road上
        :return:
        """
        pass

    def init_driver_distribution_test(self):
        """
        用于测试 每条路上初始化两个司机
        :return:
        """
        i = 0
        for road in self.roads:
            road.driver.append(Driver(i))
            i += 1
            road.driver.append(Driver(i))
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
        for i in range(self.road_nums):
            all_road_observation[i] = [len(self.roads[i].orders),
                                       len(self.roads[self.roads[i].neighbor_road[0].item()].orders),
                                       len(self.roads[self.roads[i].neighbor_road[1].item()].orders),
                                       len(self.roads[self.roads[i].neighbor_road[2].item()].orders),
                                       len(self.roads[self.roads[i].neighbor_road[3].item()].orders),
                                       len(self.roads[self.roads[i].neighbor_road[4].item()].orders),
                                       self.roads[i].traffic_congestion,
                                       self.roads[self.roads[i].neighbor_road[0].item()].traffic_congestion,
                                       self.roads[self.roads[i].neighbor_road[1].item()].traffic_congestion,
                                       self.roads[self.roads[i].neighbor_road[2].item()].traffic_congestion,
                                       self.roads[self.roads[i].neighbor_road[3].item()].traffic_congestion,
                                       self.roads[self.roads[i].neighbor_road[4].item()].traffic_congestion]

        return all_road_observation

    def assign_order(self):
        """
        分配订单
        :return:
        """
        pass

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

    def apply_policy(self, road, policy):
        """
        采取动作
        :param road:
        :param policy:
        :return:
        """
        for driver in road.driver:
            if driver.is_controllable:
                driver.next_road_id = np.argmax(policy[driver.driver_id])  # 执行动作
                driver.is_controllable = False

    def step(self, action):
        """
        执行动作，环境step
        :return:
        """
        for road in self.roads:
            self.apply_policy(road, action)

        r_n = self.assign_order()  # 进行订单分配,返回每条路上被分配的订单价格和

        self.time_slot += 1

        if self.time_slot <= 1440:
            done_n = False
        else:
            done_n = True

        obs_next = self.update_road_status()  # 更新道路状态

        return obs_next, r_n, done_n