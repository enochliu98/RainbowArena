from rainbowarena.games.ticket2ride.utils import init_board_map, init_train_car_cards, init_ticket_cards, init_lines

import numpy as np

class Ticket2RideDealer:
    def __init__(self, players, np_random=np.random.RandomState(), num_players=2):
        self.np_random = np_random  # 随机库
        
        # 【线路信息】
        self.board_map = init_board_map()  # 初始化地图
        self.lines, self.line_index, self.action_line = init_lines()
        # 【车厢卡】
        self.train_car_cards = init_train_car_cards()  # 初始化车厢卡
        # 【车票卡】
        self.ticket_cards = init_ticket_cards()  # 初始化车票卡

        self.shuffle()  # 随机洗牌

        for player in players:  # 初始化牌
            for _ in range(4):
                train_car_card = self.train_car_cards.pop()
                player.train_car_cards[train_car_card] += 1

        self.deck = self.init_deck(num_players)  # 初始化桌面上的牌 一共两部分

    def shuffle(self):
        ''' 
            随机洗牌
        '''
        self.np_random.shuffle(self.ticket_cards)
        self.np_random.shuffle(self.train_car_cards)

    def init_deck(self):
        """
            初始化桌面上的牌
        """
        deck = [[], []]

        for _ in range(5):
            deck[0].append(self.train_car_cards.pop())
        for _ in range(3):
            deck[1].append(self.ticket_cards.pop()) 

        return deck
    
    def deal_cards(self, level):
        """
            补齐空缺的牌
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

    