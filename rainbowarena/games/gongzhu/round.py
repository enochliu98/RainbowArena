import copy

from rainbowarena.games.gongzhu import Dealer


class GongzhuRound:
    def __init__(self, np_random, num_players, players):
        self.np_random = np_random
        self.dealer = Dealer(players, self.np_random)  # 初始化dealer
        self.num_players = num_players

        self.idx_1 = 0
        self.idx_2 = 52

        self.current_player = self.np_random.randint(0, self.num_players)  # 随机初始化玩家编号
        self.round_player = self.current_player  # 当前轮的庄家
        self.dun_num = 0
        self.dun_cards = []  # 每一墩开始初始化
        self.dun_color = -1  # 当前墩的颜色

    def proceed_round(self, player, action):
        """
        一共两类动作，分别对应两个阶段
        ① 加倍：0(or 16)
        ② 出牌：52
        """
        idx_1 = self.idx_1

        if action < idx_1:
            pass
        else:
            card_select = action - idx_1
            player.cards[card_select] -= 1
            self.dun_cards.append((player.player_id, card_select))
            if len(self.dun_cards) == 1:  # 当前墩的第一张
                self.dun_color = card_select % 4  # 当前墩颜色

    def get_legal_actions(self, player):
        """
        一共两类动作，分别对应两个阶段
        ① 加倍：0(or 16)
        ② 出牌：52
        """
        idx_1 = self.idx_1
        idx_2 = self.idx_2

        legal_actions = []

        for action in range(idx_1 + idx_2):
            if action < idx_1:
                pass
            else:
                card_select = action - idx_1
                if player.cards[card_select] == 1:
                    if self.dun_color == -1:  # 当前墩不限制颜色
                        legal_actions.append(action)
                    else:  # 当前墩限制颜色
                        # 首先需要判断是否包含相同颜色的卡，如果有则只能出这些
                        is_contain_same_color = False
                        for idx in range(13):
                            if player.cards[idx * 4 + self.dun_color] == 1:
                                is_contain_same_color = True
                                break
                        if is_contain_same_color is True:
                            if card_select % 4 == self.dun_color:
                                legal_actions.append(action)
                        else:
                            legal_actions.append(action)

        return legal_actions
