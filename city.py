import dgl
import numpy as np


class Road:
    def __init__(self, road_id, length, neighbor_road):
        self.id = road_id

        self.length = length  # road 长度
        self.driver = []
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
        :return:
        """
        pass

    def get_road_observation(self):
        """
        得到每条road的observation
        :return:
        """
        pass

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
        :param policy:
        :return:
        """
        for driver in road.driver:
            if driver.is_controllable:
                driver.next_road_id = np.argmax(policy[driver.driver_id])  # 执行动作
                driver.is_controllable = False

    def step(self, policy):
        """
        执行动作，环境step
        :return:
        """
        for road in self.roads:
            self.apply_policy(road, policy)

        self.time_slot += 1
