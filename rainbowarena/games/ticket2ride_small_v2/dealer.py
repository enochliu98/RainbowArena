from rainbowarena.games.ticket2ride_small_v2.utils import init_board_map, init_train_car_cards, init_ticket_cards, init_lines

import numpy as np


class Ticket2RideDealer:
    def __init__(self, players, np_random=np.random.RandomState()):
        """
            ① 初始化相关信息
            ② 洗牌
            ③ 给玩家发车厢卡
            ④ 摆牌
        """

        # 【随机信息】
        self.np_random = np_random  # 随机库
        # 【线路】
        self.board_map = init_board_map()  # 初始化地图，包含：（0）线路记录（1）地图
        self.lines, self.line_action, self.action_line = init_lines()  # 线路其他信息：线路，线路动作对应关系，动作线路对应关系
        # 【车厢卡】
        self.train_car_cards = init_train_car_cards()  # 初始化车厢卡
        # 【车票卡】
        self.ticket_cards = init_ticket_cards()  # 初始化车票卡

        self.shuffle()  # 随机洗牌

        for player in players:  # 初始化牌
            for _ in range(4):
                train_car_card = self.train_car_cards.pop()
                player.train_car_cards[train_car_card] += 1

        self.deck = self.init_deck()  # 初始化桌面上的牌 一共两部分

    def shuffle(self):
        """
            随机洗牌，主要包含两部分：
            ① 车票卡
            ② 车厢卡
        """
        self.np_random.shuffle(self.ticket_cards)
        self.np_random.shuffle(self.train_car_cards)

    def init_deck(self):
        """
            初始化桌面上的牌，主要包含两部分：
            ① 车厢卡初始化【0】
            ② 车票卡初始化【1】(不对外展示)
        """
        deck = [[], []]

        for _ in range(5):
            deck[0].append(self.train_car_cards.pop())
        for _ in range(3):
            deck[1].append(self.ticket_cards.pop())

        return deck

    def deal_cards(self, level):
        """
            补齐空缺的牌，主要包含两部分：
            ① 车厢卡摆放，如果拿走了就进行填充
            ② 车票卡摆放，如果拿走了进行重新洗牌
        """
        if level == 0:
            for i in range(5):
                if self.deck[0][i] == -1:
                    if len(self.train_car_cards) > 0:  # 牌堆中有牌才可以，如果没牌则不发
                        self.deck[0][i] = self.train_car_cards.pop()
        else:
            for ticket_card in self.deck[1]:
                self.ticket_cards.append(ticket_card)
            self.np_random.shuffle(self.ticket_cards)
            self.deck[1].clear()
            for _ in range(3):
                if len(self.ticket_cards) > 0:  # 牌堆中有牌才可以，如果没牌不发
                    self.deck[1].append(self.ticket_cards.pop())
