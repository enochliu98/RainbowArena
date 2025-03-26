"""
车票之旅：包含车票卡和车厢卡
车票卡：编号，始发站，终点站，分数
车厢卡：只有颜色区分（不需要针对性构造）
"""


class Ticket2RideTicketCard:
    def __init__(self, index=-1, start_pos=-1, end_pos=-1, point=0):
        self.index = index  # 车票卡的编号
        self.start_pos = start_pos  # 开始位置
        self.end_pos = end_pos
        self.point = point
