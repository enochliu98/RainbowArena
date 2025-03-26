from rainbowarena.games.gongzhu.utils import init_cards

import numpy as np


class GongzhuDealer:

    def __init__(self, players, np_random=np.random.RandomState()):
        """
        整局初始化
        ① 初始化牌库
        ② 发牌
        """
        self.np_random = np_random  # 随机库

        self.cards = init_cards()  # 初始化牌库里的牌
        self.shuffle()  # 随机洗牌

        for player in players:
            for _ in range(13):  # 当前为第几轮，则发几张牌
                card = self.cards.pop()
                player.cards[card] += 1  # 加入到玩家手牌

    def init_game(self, players):
        """
        一整局中小局初始化，每次都需要重新洗牌发牌
        """
        self.cards = init_cards()  # 初始化牌库里的牌
        self.shuffle()  # 随机洗牌

        for player in players:
            for _ in range(13):  # 当前为第几轮，则发几张牌
                card = self.cards.pop()
                player.cards[card] += 1  # 加入到玩家手牌

    def shuffle(self):
        """
            随机洗牌
        """
        self.np_random.shuffle(self.cards)
