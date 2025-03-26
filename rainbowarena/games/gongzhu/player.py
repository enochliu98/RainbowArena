class GongzhuPlayer:

    def __init__(self, player_id=-1):
        """
        ① 玩家编号
        ② 玩家分数
        ③ 玩家当前局手牌
        ④ 玩家当前局分数牌
        """
        self.player_id = player_id

        self.point = 0

        self.cards = [0] * 52
        self.point_cards = [0] * 52

    def init_game(self):
        """
        每个小局都需要重新初始化手牌
        """
        self.cards = [0] * 52
        self.point_cards = [0] * 52

    def compute_point(self):
        """
        根据point_cards计算分数
        """
        point = 0

        is_dmg = True if sum(self.point_cards) == 16 else False  # 判断是否大满贯

        is_qh = False  # 判断是否全红
        count_heart = 0
        for idx in range(13):
            if self.point_cards[idx * 4] == 1:
                count_heart += 1
        if count_heart == 13:
            is_qh = True

        is_only_double = False  # 是否只有变压器
        if sum(self.point_cards) == 1 and self.point_cards[43] == 1:
            is_only_double = True

        if is_dmg:
            point = 800
        elif is_qh:
            point = 200
            if self.point_cards[49] == 1:
                point -= 100
            if self.point_cards[46] == 1:
                point += 100
            if self.point_cards[43] == 1:
                point *= 2
        elif is_only_double:
            point = 50
        else:
            if self.point_cards[0] == 1:
                point -= 50
            if self.point_cards[16] == 1:
                point -= 10
            if self.point_cards[20] == 1:
                point -= 10
            if self.point_cards[24] == 1:
                point -= 10
            if self.point_cards[28] == 1:
                point -= 10
            if self.point_cards[32] == 1:
                point -= 10
            if self.point_cards[36] == 1:
                point -= 10
            if self.point_cards[40] == 1:
                point -= 20
            if self.point_cards[44] == 1:
                point -= 30
            if self.point_cards[48] == 1:
                point -= 40
            if self.point_cards[49] == 1:
                point -= 100
            if self.point_cards[46] == 1:
                point += 100
            if self.point_cards[43] == 1:
                point *= 2

        return point
