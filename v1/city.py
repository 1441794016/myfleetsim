import dgl
import numpy as np
import random
import torch
from driver import Driver
from order import Order


class Road:
    def __init__(self, road_id, length, neighbor_road):
        self.id = road_id

        self.length = length  # road 长度
        self.drivers = []  # 该road上的所有driver
        # self.controllable_driver = []  # 存放该road上可以接受导航决策车辆
        # self.uncontrollable_driver = []  # 存放该road上不需要进行导航决策的车辆
        self.orders = []  # 该road上出现过的所有order，包括待接收，已接受和已过期的订单
        self.traffic_congestion = 10  # 一个道路拥挤指标
        self.neighbor_road = neighbor_road  # 存放该road 在 n=1 跳内可到达的road

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

        road_income = 0
        number_of_pair_has_been_matched = 0
        for driver_index in range(len(self.drivers)):
            if self.drivers[driver_index].is_serving is False:
                for order_index in range(len(self.orders)):
                    if self.orders[order_index].is_accepted is False and self.orders[order_index].is_overdue is False:
                        self.drivers[driver_index].order_match(self.orders[order_index])
                        self.orders[order_index].driver_match(self.drivers[driver_index])

                        road_income += self.orders[order_index].price
                        number_of_pair_has_been_matched += 1
                        break
            if number_of_pair_has_been_matched >= assignable_number:
                break

        # 对orders和driver列表重新按id序号进行排序
        self.orders.sort(key=lambda x: x.id)
        self.drivers.sort(key=lambda x: x.id)

        return road_income


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
            neigh = []
            index = 0
            for node_index in self.road_graph.edges()[0]:
                if node_index.item() == i:
                    neigh.append(self.road_graph.edges()[1][index].item())
                index += 1

            if len(neigh) < 5:
                for j in range(5 - len(neigh)):
                    neigh.append(float('-inf'))

            length = self.road_graph.ndata['length'][i]  # tensor类型
            # frontier = self.road_graph.sample_neighbors(i, -1)  # 查询节点i的所有一阶邻居 有误！ 实际上得到的是是指向该点的点
            # neighbor_road = frontier.edges()[0]  # tensor类型
            self.roads.append(Road(road_id=road_id, length=length, neighbor_road=neigh))

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
        在road 1 处初始化8个订单
        在road 2 处初始化4个订单
        :return:
        """
        for i in range(12):
            if i <= 7:
                self.roads[1].orders.append(Order(i))
            else:
                self.roads[2].orders.append(Order(i))

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
        observation = np.zeros((self.road_nums, 12))
        for i in range(self.road_nums):
            inf_count = self.roads[i].neighbor_road.count(float('-inf'))
            for j in range(6):
                if j < 6 - inf_count:
                    if j == 0:
                        observation[i][j] = len([order for order in self.roads[i].orders if order.is_accepted is False])
                        observation[i][j + 6] = self.roads[i].traffic_congestion
                    else:
                        observation[i][j] = len([order for order in self.roads[self.roads[i].neighbor_road[j - 1]].orders \
                                                if order.is_accepted is False])
                        observation[i][j + 6] = self.roads[self.roads[i].neighbor_road[0]].traffic_congestion
                else:
                    observation[i][j] = 0
                    observation[i][j + 6] = 0

        return observation

            # all_road_observation[i] = [len([order for order in self.roads[i].orders if order.is_accepted is False]),
            #                            len([order for order in self.roads[self.roads[i].neighbor_road[0]].orders \
            #                                 if order.is_accepted is False]),
            #                            len([order for order in self.roads[self.roads[i].neighbor_road[1]].orders \
            #                                 if order.is_accepted is False]),
            #                            len([order for order in self.roads[self.roads[i].neighbor_road[2]].orders \
            #                                 if order.is_accepted is False]),
            #                            len([order for order in self.roads[self.roads[i].neighbor_road[3]].orders \
            #                                 if order.is_accepted is False]),
            #                            len([order for order in self.roads[self.roads[i].neighbor_road[4]].orders \
            #                                 if order.is_accepted is False]),
            #                            self.roads[i].traffic_congestion,
            #                            self.roads[self.roads[i].neighbor_road[0].item()].traffic_congestion,
            #                            self.roads[self.roads[i].neighbor_road[1].item()].traffic_congestion,
            #                            self.roads[self.roads[i].neighbor_road[2].item()].traffic_congestion,
            #                            self.roads[self.roads[i].neighbor_road[3].item()].traffic_congestion,
            #                            self.roads[self.roads[i].neighbor_road[4].item()].traffic_congestion]

        #return all_road_observation

    def assign_order(self):
        """
        Randomly assign call to the drivers at the same road.
        对每一条road都分配订单
        :return:
        """
        total_income = np.zeros(self.road_nums)  # 总收入,np数组
        for road_index in range(self.road_nums):
            total_income[road_index] = self.roads[road_index].order_driver_match()
        return total_income

    def update_driver_status(self):
        """
        采取动作后将driver的状态更新为下一个时隙开始时的状态
        暂时假设在时隙开始时已经给出将要去的
        :return:
        """
        for road_index in range(self.road_nums):
            for driver_index in range(len(self.roads[road_index].drivers)):
                if self.roads[road_index].drivers[driver_index].is_serving is False and \
                        self.roads[road_index].drivers[driver_index].next_road is not None:  # 对被给出调度决策的车进行状态转移
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
        for road_index in range(self.road_nums):
            j = 0
            for driver_index in range(len(self.roads[road_index].drivers)):
                driver = self.roads[road_index].drivers[driver_index]
                if driver.is_serving is False and driver.next_road is None:  # 对没接单且没有做出路径决策的车执行动作
                    next_road = np.random.choice(self.roads[road_index].neighbor_road, p=action[i][j])
                    self.roads[road_index].drivers[driver_index].next_road = next_road
                    j += 1
            i += 1

    def step(self, action):
        """
        执行动作，环境step
        :return:
        """
        self.apply_action(action)

        if self.time_slot <= 1440:
            done_n = False
        else:
            done_n = True

        self.update_driver_status()
        self.update_road_status()
        r = self.assign_order()  # 进行订单分配,返回每条路上被分配的订单价格和
        obs_next = self.get_road_observation()  # 更新道路状态
        self.time_slot += 1
        return obs_next, r, done_n
