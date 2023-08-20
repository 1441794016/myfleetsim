class Driver:
    def __init__(self, driver_id):
        self.id = driver_id
        self.is_online = True
        self.located_road = 0  # 该driver所在的road编号
        self.is_controllable = True  # 是否由agent给出下一个要去的道路 即目前是否可以控制
        self.is_serving_order = False  # 是否接受了order
        self.next_road_index = 0  # 如果agent给出了决策，表示将要去的下一条路；0表示留在本条路


