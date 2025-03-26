import copy

from rainbowarena.games.papayoo import Dealer


class PapayooRound:
    def __init__(self, np_random, num_players, players):
        self.np_random = np_random
        self.dealer = Dealer(players, self.np_random)  # 初始化dealer
        self.num_players = num_players

        self.current_player = self.np_random.randint(0, self.num_players)  # 随机初始化玩家编号
        self.round_player = self.current_player  # 当前轮的庄家
        self.round_num = 0
        self.dun_num = 0
        self.dun_cards = []  # 每一墩开始初始化
        self.dun_color = -1  # 当前墩的颜色

    def proceed_round(self, player, action):
        """
        一类动作：出牌（60）
        """

        card_select = action
        player.cards[card_select] -= 1
        self.dun_cards.append((player.player_id, card_select))
        if len(self.dun_cards) == 1:  # 当前墩的第一张
            self.dun_color = card_select % 4 if card_select < 40 else 4  # 当前墩颜色,0~3代表四花色，4代表payo花色

    def get_legal_actions(self, player):
        """
        一类动作：出牌（60）
        """
        legal_actions = []

        # 判断是否包含相同颜色
        is_contain_same_color = False
        for card_tmp in range(60):
            if player.cards[card_tmp] == 1:  # 有这张牌
                card_tmp_color = card_tmp % 4 if card_tmp < 40 else 4
                if card_tmp_color == self.dun_color:
                    is_contain_same_color = True
                    break

        for action in range(60):
            card_select = action
            if player.cards[card_select] == 1:  # 有这张牌
                if self.dun_color == -1:  # 当前墩不限制颜色
                    legal_actions.append(action)
                else:  # 当前墩限制颜色
                    if is_contain_same_color is True:  # 包含则必须出相同颜色的
                        card_color = card_select % 4 if card_select < 40 else 4
                        if card_color == self.dun_color:
                            legal_actions.append(action)
                    else:  # 不包含随便出
                        legal_actions.append(action)

        return legal_actions
