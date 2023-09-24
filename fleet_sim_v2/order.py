class Order:
    def __init__(self, order_id, price=700, start_road=None, end_road=None):
        self.id = order_id
        self.is_accepted = False  # 该订房是否已被driver接受的标志 False表示还没有被接受
        self.is_overdue = False  # 该订单是否因为长久未被接受而过期 False表示还未过期
        self.served_driver = None  # 接受该订单的driver，未被接受（即is_accepted is False）则为None，否则为Driver类型
        self.price = price  # 该订单的价格
        self.start_road = start_road  # 订单的始发地是哪条road
        self.end_road = end_road  # 订单的目的地是哪条road

    def driver_match(self, driver):
        """
        将该order与参数中的driver进行匹配
        :param driver:
        :return:
        """
        self.served_driver = driver
        self.is_accepted = True
