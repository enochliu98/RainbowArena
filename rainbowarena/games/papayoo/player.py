class PapayooPlayer:

    def __init__(self, player_id=-1):
        """
        ① 玩家编号
        ② 玩家分数
        ③ 玩家当前局手牌
        """
        self.player_id = player_id
        self.point = 0

        self.cards = [0] * 60

    def init_game(self):
        """
        每个小局都需要重新初始化手牌
        """
        self.cards = [0] * 60

