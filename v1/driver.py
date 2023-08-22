class Driver:
    def __init__(self, driver_id):
        self.id = driver_id
        self.is_serving = False  # 该driver是否已经匹配里订单 且还没完成订单
        self.serving_order = None  # 目前正在服务的订单，没有则为None，有则为Order类型
        self.destination = None  # 目的地，只有再serving_order不为None时才有效，实际上是所接受订单的目的地，用road编号表示

        self.now_location_road = None  # 目前所在哪条road, 用road编号表示
        self.next_road = None  # 下一个要去的临近的road，为None表示还没有给出决策，否则用road编号表示

    def order_match(self, order):
        self.serving_order = order
        self.is_serving = True
        self.destination = order.end_road
