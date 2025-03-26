class SplendorCard:
    def __init__(self, card_type=-1, cost=[0, 0, 0, 0, 0], point=0, bonus=0):
        ''' Initialize the class of Card

        Args:
            card_type (int): 第几级别
            cost (int): 购买该卡需要消耗的宝石
            point (int): 购买该卡得到的分数
            bonus (int): 购买该卡可获得的宝石抵扣
        '''
        
        self.card_type = card_type
        self.cost = cost
        self.point = point
        self.bonus = bonus

        

