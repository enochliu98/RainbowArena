from rainbowarena.games.century.utils import init_function_cards, init_point_cards

import numpy as np
import copy

# 开始阶段发香料
deal_tokens_init = [
    [3, 0, 0, 0],
    [4, 0, 0, 0],
    [4, 0, 0, 0],
    [3, 1, 0, 0],
    [3, 1, 0, 0]
]


class CenturyDealer:

    def __init__(self, players, current_player, np_random=np.random.RandomState(), num_players=2):
        """
            ① 随机库
            ② 功能卡
            ③ 分数卡
            ④ 发牌
            ⑤ 洗牌
            ⑥ 摆牌
        """
        # 【随机库】
        self.np_random = np_random
        # 【功能卡】
        self.function_cards = init_function_cards()
        self.all_function_cards = copy.deepcopy(self.function_cards)  # 功能卡副本
        # 【分数卡】
        self.point_cards = init_point_cards()
        # 【发牌】
        self.deal_tokens_cards_init(players, current_player)
        # 【洗牌】
        self.shuffle()
        # 【摆牌】
        self.deck = self.init_deck(num_players)

    def deal_tokens_cards_init(self, players, current_player):
        """
            发牌
            ① 发香料
            ② 发功能卡
        """
        num_players = len(players)
        for i in range(num_players):
            player_id = (current_player + i) % num_players
            for token_idx in range(4):
                players[player_id].tokens[token_idx] = deal_tokens_init[i][token_idx]
            players[player_id].function_cards.append(self.function_cards[i])  # 一张升级2
            players[player_id].function_cards.append(self.function_cards[i + 40])  # 一张拿2黄
            players[player_id].map_function_cards[i] = 0
            players[player_id].map_function_cards[i + 40] = 1

        self.function_cards = self.function_cards[num_players:40] + self.function_cards[40 + num_players:]

    def shuffle(self):
        """
            随机洗牌
            ① 功能卡
            ② 分数卡
        """
        self.np_random.shuffle(self.function_cards)
        self.np_random.shuffle(self.point_cards)

    def init_deck(self, num_players):
        """
            初始化桌面上的牌
            ① 功能卡
            ② 功能卡上面的方块数目
            ③ 分数卡
            ④ 分数卡上面的金银币分数
        """
        deck = [[], [], [], []]

        for _ in range(6):
            deck[0].append(self.function_cards.pop())  # 功能牌分发
            deck[1].append([0] * 4)  # 功能牌上面对应的各种方块数目
        for _ in range(5):
            deck[2].append(self.point_cards.pop())  # 分数牌分发
        deck[3].extend([3, 1, 0, 0, 0])  # 金币银币分发

        return deck

    def deal_cards(self, level, position):
        """
            补齐空缺的牌
        """
        if level == 0:
            self.deck[level][position:-1] = self.deck[level][position + 1:]
            if len(self.function_cards) > 0:  # 还有功能卡
                self.deck[level][-1] = self.function_cards.pop()
            else:
                self.deck[level][-1] = -1  # 默认值

            self.deck[level + 1][position:-1] = self.deck[level + 1][position + 1:]
            self.deck[level + 1][-1] = [0] * 4
        else:
            self.deck[level][position:-1] = self.deck[level][position + 1:]
            self.deck[level][-1] = self.point_cards.pop()
