
class Order:
    def __init__(self, order_id):
        self.id = order_id
        self.start_time = 0
        self.end_time = 0
        self.start_location = 0  # 提供服务的起始地点
        self.end_location = 0  # 服务的目的地
        self.price = 0
        self.is_accepted = False
