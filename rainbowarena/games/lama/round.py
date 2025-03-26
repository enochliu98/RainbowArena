import copy

from rainbowarena.games.lama import Dealer


class LamaRound:
    def __init__(self, np_random, num_players, players):
        self.np_random = np_random  # 随机库
        self.dealer = Dealer(players, self.np_random)  # 初始化dealer，每小局都需要更新
        self.num_players = num_players  # 玩家数目
        self.current_player = self.np_random.randint(0, self.num_players)  # 初始化玩家编号，每小局都需要更新

    def proceed_round(self, player, action):
        """
        一共三类动作:
        ① 出牌 7
        ② 抽牌 1
        ③ 放弃 1
        """
        if action < 7:  # 动作1-出牌
            player.cards[action] -= 1  # 出对应的牌
            self.dealer.deck = action  # 牌顶变换
        elif action < 8:  # 动作2-抽牌
            card = self.dealer.cards.pop()
            player.cards[card] += 1  # 抽牌
        else:  # 动作3-放弃
            player.quit = 1  # 放弃

    def get_legal_actions(self, player, players):
        """
        一共三类动作:
        ① 出牌 7
        ② 抽牌 1
        ③ 放弃 1
        """
        legal_actions = []

        # 除自己之外的其他人是否都退出
        is_others_quit = True
        for p in players:
            if p.player_id != player.player_id:
                if p.quit == 0:  # 有人没退出
                    is_others_quit = False
                    break

        for action in range(9):
            if action < 7:  # 动作1-出牌
                if player.cards[action] > 0:
                    if action == self.dealer.deck or action == (self.dealer.deck + 1) % 7:  # 满足牌的要求
                        legal_actions.append(action)
            elif action < 8:  # 动作2-抽牌
                if len(self.dealer.cards) > 0 and is_others_quit is False:  # 有牌且有人没退出
                    legal_actions.append(action)
            else:  # 动作3-放弃
                legal_actions.append(action)

        return legal_actions
