'''
车票之旅：包含车票卡和车厢卡
车票卡：始发站，终点站，分数
车厢卡：只有颜色区分
'''


class Ticket2RideTicketCard:
    def __init__(self, start_pos=-1, end_pos=-1, point=0):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.point = point
