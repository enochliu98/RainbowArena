"""
在香料之路中, 一共有两种类型的卡片
【功能卡(商人卡)】: FunctionCard
功能卡只需要关注他的bonus即可,bonus主要指转换关系
type
0  香料 4维
1  升级 1维
2  交易 8维
【分数卡】: PointCard
分数卡只需要关注它的cost, point即可
"""


class CenturyFunctionCard:
    """
        功能卡：
        ① 编号：1维
        ② 类型：1维
        ③ 功能：1，4，8维
    """
    def __init__(self, card_id=-1, card_type=-1, card_bonus=-1):
        self.card_id = card_id
        self.type = card_type
        self.bonus = card_bonus


class CenturyPointCard:
    """
        分数卡：
        ① 花费：4维
        ② 分数：1维
    """
    def __init__(self, card_cost=-1, card_point=-1):
        self.cost = card_cost
        self.point = card_point
