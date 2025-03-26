from rainbowarena.games.papayoo.utils import init_cards

import numpy as np


class PapayooDealer:

    def __init__(self, players, np_random=np.random.RandomState()):
        """
        整局初始化
        ① 初始化牌库
        ② 发牌
        """
        self.np_random = np_random  # 随机库

        self.cards = init_cards()  # 初始化牌库里的牌
        self.shuffle()  # 随机洗牌

        num_cards = 60 // len(players)  # 每个人应该发的牌数
        for player in players:
            for _ in range(num_cards):
                card = self.cards.pop()
                player.cards[card] += 1  # 加入到玩家手牌

        self.papayoo_color = self.np_random.randint(4)  # 随机生成颜色
        self.papayoo_card = 6 * 4 + self.papayoo_color

    def init_game(self, players):
        """
        一整局中小局初始化，每次都需要重新洗牌发牌
        """
        self.cards = init_cards()  # 初始化牌库里的牌
        self.shuffle()  # 随机洗牌

        num_cards = 60 // len(players)  # 每个人应该发的牌数
        for player in players:
            for _ in range(num_cards):  # 当前为第几轮，则发几张牌
                card = self.cards.pop()
                player.cards[card] += 1  # 加入到玩家手牌

        self.papayoo_color = self.np_random.randint(4)  # 随机生成颜色
        self.papayoo_card = 6 * 4 + self.papayoo_color

    def shuffle(self):
        """
            随机洗牌
        """
        self.np_random.shuffle(self.cards)
