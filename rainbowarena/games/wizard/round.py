import copy

from rainbowarena.games.wizard import Dealer


class WizardRound:
    def __init__(self, np_random, num_players, players):
        self.np_random = np_random
        self.dealer = Dealer(players, self.np_random)  # 初始化dealer
        self.num_players = num_players

        self.idx_1 = 4
        self.idx_2 = 60 // num_players + 1  # 60//3=20 60//4=15 60//5=12 60//6=10
        self.idx_3 = 60

        self.round_num = 0  # 已完成的round数目
        self.dun_num = 0  # 已完成的dun数目
        self.current_player = self.np_random.randint(0, self.num_players)  # 随机初始化玩家编号
        self.round_player = self.current_player  # 当前轮的发牌玩家
        self.dun_cards = []  # 每一墩开始初始化
        self.dun_color = -1

    def proceed_round(self, player, action):
        """
        一共三类动作，分别对应三个阶段 (4+16+60=80)
        动作1：选王牌花色 (4) 与玩家数目无关
        动作2：预测墩数 (16) 与玩家数目有关
        动作3：出牌 (60) 与玩家数目无关
        """
        idx_1 = self.idx_1
        idx_2 = self.idx_2

        if action < idx_1:
            self.dealer.ace_color = action
        elif action < idx_1 + idx_2:
            player.duns_prd = action - idx_1
        else:
            card_select = action - (idx_1 + idx_2)
            player.cards[card_select] -= 1
            self.dun_cards.append((player.player_id, card_select))
            if self.dun_color == -1:  # 还没确定当前墩颜色
                if card_select < 52:
                    self.dun_color = card_select % 4
                elif card_select < 56:
                    self.dun_color = 4
                else:
                    self.dun_color = -1

    def get_legal_actions(self, player):
        """
        一共三类动作，分别对应三个阶段 (4+16+60=80)
        动作1：选王牌花色 (4)
        动作2：预测墩数 (16)
        动作3：出牌 (60)
        """
        idx_1 = self.idx_1
        idx_2 = self.idx_2
        idx_3 = self.idx_3

        legal_actions = []

        for action in range(idx_1 + idx_2 + idx_3):
            if action < idx_1:
                if self.dealer.ace_color == -1:  # 还没确定王牌花色
                    legal_actions.append(action)
            elif action < idx_1 + idx_2:
                if self.dealer.ace_color != -1:  # 确定王牌花色
                    if player.duns_prd == -1 and action - idx_1 <= sum(player.cards):
                        legal_actions.append(action)
            else:
                card_select = action - (idx_1 + idx_2)
                if self.dealer.ace_color != -1 and player.duns_prd != -1:  # 确定了王牌花色和墩数
                    if self.dun_color == -1 or self.dun_color == 4:  # 当前墩还没确定颜色或已经有人出巫师
                        if player.cards[card_select] == 1:
                            legal_actions.append(action)
                    else:  # 当前墩已经确定颜色
                        if card_select < 52:
                            # 先判断手牌里是否有墩颜色一样的颜色牌
                            is_same_dun_color = False
                            for card_idx in range(13):
                                if player.cards[card_idx * 4 + self.dun_color] == 1:
                                    is_same_dun_color = True
                                    break
                            if is_same_dun_color is True:
                                if player.cards[card_select] == 1 and card_select % 4 == self.dun_color:
                                    legal_actions.append(action)
                            else:
                                if player.cards[card_select] == 1:
                                    legal_actions.append(action)
                        else:
                            if player.cards[card_select] == 1:
                                legal_actions.append(action)

        return legal_actions
