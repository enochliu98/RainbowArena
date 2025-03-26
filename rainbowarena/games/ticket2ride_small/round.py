# -*- coding: utf-8 -*-

import functools
import numpy as np
import copy

from rainbowarena.games.ticket2ride_small import Dealer

TRAIN_CAR_CARDS_AVAILABLE = [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
    [1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1]
]
POINTS = {1: 1,
          2: 2,
          3: 4,
          }

TICKET_CARDS_AVAILABLE = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
]


class Ticket2RideRound:
    ''' Round can call other Classes' functions to keep the game running
    '''

    def __init__(self, np_random, num_players, players):
        self.np_random = np_random
        self.dealer = Dealer(self.np_random, num_players)  # 初始化dealer
        self.num_players = num_players
        self.current_player = self.np_random.randint(0, self.num_players)  # 随机初始化玩家编号
        self.init_step = True  # 初始阶段标志，在初始阶段，所有玩家都先进行抽车票卡

    def proceed_round(self, player, action):
        # 动作空间:
        # 1.火车车厢牌 5选2+5选1 15
        # 2.声明线路 12 + 5*4 = 32
        # 3.摸目的地车牌 3选1+3选2+3选3 7 

        if action < 15:  # [火车车厢牌]
            train_car_card_selected = TRAIN_CAR_CARDS_AVAILABLE[action]  # 可选的车厢卡

            # 【1】player从dealer获取车厢卡
            for i in range(5):
                if train_car_card_selected[i] == 1:
                    player.train_car_cards[self.dealer.deck[0][i]] += 1  # 玩家获取
                    self.dealer.deck[0][i] = -1  # dealer更新
            # 【2】dealer补充车厢卡
            self.dealer.deal_cards(0)

        elif action < 47:  # [声明线路]
            line_selected = action - 15
            line_selected = self.dealer.action_line[line_selected]
            line_info = self.dealer.board_map[0][line_selected]  # 所选线路的信息

            # 【1】dealer更新线路占用情况
            if self.num_players <= 3:
                lines_set = self.dealer.board_map[1][line_info[3]][line_info[4]]  # 如果对应位置有两条线路，对于两人游戏都需要进行占用
                for line in lines_set:
                    self.dealer.board_map[0][line][5] = 1  # 将该线路设置为被占用
            else:
                self.dealer.board_map[0][line_info[0]][5] = 1  # 将该线路设置为被占用

            # 【2】player更新线路，车厢信息
            # 【3】dealer更新车厢信息      
            player.line_map_valid[line_info[0]] = 1  # 目前已经连通的线路
            player.train_cars -= line_info[1]  # 将火车车厢摆上去

            if line_info[2] == 5:
                train_car_type = (action - 15) - self.dealer.line_index[line_selected]  # 不限制颜色
            else:
                train_car_type = line_info[2]  # 限制颜色

            if player.train_car_cards[train_car_type] >= line_info[1]:  # 够用
                player.train_car_cards[train_car_type] -= line_info[1]  # 付出对应的车厢卡
                for _ in range(line_info[1]):
                    self.dealer.train_car_cards.append(train_car_type)  # 把卡还给dealer
            else:  # 不够用，需要使用火车头
                train_car_cards_1 = player.train_car_cards[train_car_type]  # 现有的
                train_car_cards_2 = line_info[1] - player.train_car_cards[train_car_type]  # 需要使用通用卡片的
                player.train_car_cards[train_car_type] -= train_car_cards_1  # 付出对应的车厢卡
                for _ in range(train_car_cards_1):
                    self.dealer.train_car_cards.append(train_car_type)
                player.train_car_cards[0] -= train_car_cards_2  # 付出通用车厢卡
                for _ in range(train_car_cards_2):
                    self.dealer.train_car_cards.append(0)
            self.np_random.shuffle(self.dealer.train_car_cards)  # 重新洗牌
            self.dealer.deal_cards(0)  # 发牌
            player.longest_line = line_info[1] if line_info > player.longest_line else player.longest_line  # 最长线路更新
            player.board_map_valid[line_info[3]][line_info[4]] = 1  # 更新无向图
            player.board_map_valid[line_info[4]][line_info[3]] = 1  # 更新无向图

            # 【4】player更新分数信息
            player.point += POINTS[line_info[1]]  # 加分

        else:  # [摸目的地车牌]
            ticket_card_selected = TICKET_CARDS_AVAILABLE[action - 47]
            deck_tmp = copy.deepcopy(self.dealer.deck[1])

            # 【1】player更新车票信息
            # 【2】dealer移除车票
            for i in range(3):
                if ticket_card_selected[i] == 1:
                    player.ticket_cards.append(deck_tmp[i])  # 添加指定元素
                    self.dealer.deck[1].remove(deck_tmp[i])  # 删除指定元素

            # 【3】dealer补充车票
            self.dealer.deal_cards(1)

    def get_legal_actions(self, player):
        legal_actions = []  # 可用动作列表

        if self.init_step is True:
            legal_actions = [50, 51, 52, 53]
            self.init_step = False
        else:
            for action in range(54):
                if action < 15:  # [火车车厢牌]
                    train_car_card_selected = TRAIN_CAR_CARDS_AVAILABLE[action]  # 可选的车厢卡
                    if action < 5:
                        for i in range(5):
                            if train_car_card_selected[i] == 1 and self.dealer.deck[0][i] == 0:
                                legal_actions.append(action)
                                break
                    else:
                        is_available = True
                        for i in range(5):
                            if train_car_card_selected[i] == 1 and self.dealer.deck[0][i] == -1:  # 没牌
                                is_available = False
                                break
                            if train_car_card_selected[i] == 1 and self.dealer.deck[0][i] == 0:  # 是万能牌
                                is_available = False
                                break
                        if is_available is True:
                            legal_actions.append(action)
                elif action < 47:  # [声明线路]
                    line_selected = action - 15
                    line_selected = self.dealer.action_line[line_selected]
                    line_info = self.dealer.board_map[0][line_selected]  # 所选线路的信息

                    is_available = True

                    if line_info[2] == 5:
                        train_car_type = (action - 15) - self.dealer.line_index[line_selected]  # 不限制颜色
                    else:
                        train_car_type = line_info[2]  # 限制颜色

                    if player.train_cars < line_info[1]:  # 现有车厢不够
                        is_available = False
                    if train_car_type != 0 and player.train_car_cards[train_car_type] + player.train_car_cards[0] < \
                            line_info[1]:  # 车厢卡不够
                        is_available = False
                    if train_car_type == 0 and player.train_car_cards[0] < line_info[1]:
                        is_available = False
                    if line_info[5] == 1:  # 已经被占用
                        is_available = False

                    if is_available:
                        legal_actions.append(action)
                else:  # [抽车票]
                    if len(self.dealer.deck[1]) == 3:
                        legal_actions.append(action)
                    elif len(self.dealer.deck[1]) == 2:
                        if action in [47, 48, 50]:
                            legal_actions.append(action)
                    elif len(self.dealer.deck[1]) == 1:
                        if action == 47:
                            legal_actions.append(action)

        return legal_actions
