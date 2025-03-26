class GizmosCard:
    def __init__(self, card_type=-1, card_property=-1, cost=[0, 0, 0, 0], point=0, bonus=0):
        ''' Initialize the class of UnoCard

        Args:
            card_type (int): 哪一级别的卡片 (0-2)
            card_property (int) : 当前卡片是什么属性的卡片 (0-3) 黑，蓝，红，黄
            cost (int): 当前卡的消耗的资源
            point (int): 当前卡对应的分数
            bonus (int): 当前卡的效果
        '''
        
        self.card_type = card_type
        self.card_property = card_property
        self.cost = cost
        self.point = point
        self.bonus = bonus

        

