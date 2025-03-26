# -*- coding: utf-8 -*-

import functools
import numpy as np

from rainbowarena.games.splendor import Dealer

# from dealer import SplendorDealer as Dealer

FETCH_DIFF_TOKENS = [[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3],
                     [0, 3, 4], [0, 2, 4], [0, 2, 3],
                     [0, 1, 4], [0, 1, 3],
                     [0, 1, 2]]


class SplendorRound:
    ''' Round can call other Classes' functions to keep the game running
    '''

    def __init__(self, np_random, num_players):
        self.np_random = np_random
        self.dealer = Dealer(self.np_random, num_players)  # 初始化dealer
        self.num_players = num_players
        self.current_player = self.np_random.randint(0, self.num_players)  # 随机初始化玩家编号

    def proceed_round(self, player, action):
        ''' Call other Classes's functions to keep one round running

        Args:
            player (object): object of DoudizhuPlayer
            action (str): string of legal specific action

        Returns:
            object of DoudizhuPlayer: player who played current biggest cards.
        '''
        # action (5+10)+12+12=39
        if action < 5:  # 第一类动作，拿相同颜色宝石 【0，4】
            player.current_tokens[action] += 2
            self.dealer.tokens[action] -= 2
            if sum(player.current_tokens) > 10:
                discard_tokens = self.get_discard_tokens(player)
                for i in range(6):
                    player.current_tokens[i] -= discard_tokens[i]
        elif action < 15:  # 第一类动作，拿不同颜色的宝石 【5,14】
            if self.dealer.tokens[FETCH_DIFF_TOKENS[action - 5][0]] > 0:
                player.current_tokens[FETCH_DIFF_TOKENS[action - 5][0]] += 1
                self.dealer.tokens[FETCH_DIFF_TOKENS[action - 5][0]] -= 1
            if self.dealer.tokens[FETCH_DIFF_TOKENS[action - 5][1]] > 0:
                player.current_tokens[FETCH_DIFF_TOKENS[action - 5][1]] += 1
                self.dealer.tokens[FETCH_DIFF_TOKENS[action - 5][1]] -= 1
            if self.dealer.tokens[FETCH_DIFF_TOKENS[action - 5][2]] > 0:
                player.current_tokens[FETCH_DIFF_TOKENS[action - 5][2]] += 1
                self.dealer.tokens[FETCH_DIFF_TOKENS[action - 5][2]] -= 1

            if sum(player.current_tokens) > 10:
                discard_tokens = self.get_discard_tokens(player)
                for i in range(6):
                    player.current_tokens[i] -= discard_tokens[i]
                    self.dealer.tokens[i] += discard_tokens[i]
        elif action < 27:  # 第二类动作，买牌 【15,26】
            # 计算牌的位置
            level = (action - 15) // 4 + 1
            pos = (action - 15) % 4

            card = self.dealer.deck[level][pos]  # 获取牌桌上的牌
            for i, token_cost in enumerate(card.cost):
                token_cost_minus_current_card_tokens = max(token_cost - player.current_card_tokens[i], 0)
                player_cost_tokens_1 = min(player.current_tokens[i], token_cost_minus_current_card_tokens)
                player_cost_tokens_2 = max(token_cost_minus_current_card_tokens - player.current_tokens[i], 0)

                player.current_tokens[i] -= player_cost_tokens_1  # 去掉玩家宝石，当前卡的cost减去bonus宝石
                self.dealer.tokens[i] += player_cost_tokens_1  # 增加牌桌上的宝石
                player.current_tokens[-1] -= player_cost_tokens_2  # 去掉玩家万能宝石
                self.dealer.tokens[-1] += player_cost_tokens_2  # 增加牌桌上的万能宝石

            self.dealer.deal_cards(level, pos)  # 发牌补充
            player.current_cards.append(card)  # 更新玩家手牌
            player.point += card.point  # 更新玩家积分
            player.current_card_tokens[card.bonus] += 1  # 更新宝石矿卡对应宝石

        elif action < 30:  # 第二类动作，买盖着的牌 【27，29】
            pos = action - 27

            card = player.current_flip_cards[pos]  # 获取玩家盖着的牌

            for i, token_cost in enumerate(card.cost):
                token_cost_minus_current_card_tokens = max(token_cost - player.current_card_tokens[i], 0)
                player_cost_tokens_1 = min(player.current_tokens[i], token_cost_minus_current_card_tokens)
                player_cost_tokens_2 = max(token_cost_minus_current_card_tokens - player.current_tokens[i], 0)

                player.current_tokens[i] -= player_cost_tokens_1  # 去掉玩家宝石，当前卡的cost减去bonus宝石
                self.dealer.tokens[i] += player_cost_tokens_1  # 增加牌桌上的宝石
                player.current_tokens[-1] -= player_cost_tokens_2  # 去掉玩家万能宝石
                self.dealer.tokens[-1] += player_cost_tokens_2  # 增加牌桌上的万能宝石

            player.current_flip_cards.pop(pos)  # 去掉
            player.current_cards.append(card)  # 更新玩家手牌
            player.point += card.point  # 更新玩家积分
            player.current_card_tokens[card.bonus] += 1  # 更新宝石矿卡对应宝石

        elif action < 42:  # 第三类动作，盖牌 【30,41】
            # 计算牌的位置
            level = (action - 30) // 4 + 1
            pos = (action - 30) % 4

            card = self.dealer.deck[level][pos]  # 获取牌桌上的牌
            self.dealer.deal_cards(level, pos)  # 发牌补充
            player.current_flip_cards.append(card)  # 盖上对应的牌

            player.current_tokens[-1] += 1
            self.dealer.tokens[-1] -= 1

        for noble_card in self.dealer.deck[0]:
            flag = 1  # 初始化标志,1代表匹配成功，0代表匹配失败
            for i in range(5):
                if player.current_card_tokens[i] < noble_card.cost[i]:
                    flag = 0  # 设置标志
                    break
            if flag:
                player.current_noble_cards.append(noble_card)  # 更新玩家贵族卡
                self.dealer.deck[0].remove(noble_card)  # 删除牌桌上的贵族卡
                player.point += noble_card.point  # 更新玩家积分
                break  # 只要找到一张就退出

    def get_discard_tokens(self, player):
        """
            寻找要删掉的牌
        """
        best_card = None
        min_dis = 1000
        # 先检查牌面上的牌
        for level in range(1, 4):
            for pos in range(4):
                dis = 0  # 初始化距离
                card = self.dealer.deck[level][pos]  # 当前卡
                card_cost = [max(card.cost[i] - player.current_card_tokens[i], 0) for i in range(5)]  # 卡的cost
                if sum(card_cost) > 10:  # 如果大于10，则目前考虑该卡没有意义
                    continue
                for i in range(5):
                    dis += max(card_cost[i] - player.current_card_tokens[i], 0)  # 如果不够，则计数
                dis = dis - player.current_card_tokens[-1]  # 最终的距离，这里加不加都可以

                if dis < min_dis:
                    min_dis = dis
                    best_card = card
                elif dis == min_dis:
                    if best_card.point < card.point:
                        best_card = card

        for card in player.current_flip_cards:
            dis = 0
            card_cost = [max(card.cost[i] - player.current_card_tokens[i], 0) for i in range(5)]  # 卡的cost
            if sum(card_cost) > 10:  # 如果大于10，则目前考虑该卡没有意义
                continue
            for i in range(5):
                dis += max(card_cost[i] - player.current_card_tokens[i], 0)  # 如果不够，则计数
            dis = dis - player.current_card_tokens[-1]  # 最终的距离，这里加不加都可以

            if dis < min_dis:
                min_dis = dis
                best_card = card
            elif dis == min_dis:
                if best_card.point < card.point:
                    best_card = card

        best_card_cost = [max(best_card.cost[i] - player.current_card_tokens[i], 0) for i in range(5)]  # 卡的cost
        num_discard_tokens = sum(player.current_tokens) - 10  # 需要丢弃的token的数目
        discard_tokens = [0, 0, 0, 0, 0]  # 具体需要丢弃的宝石

        # 第一遍
        start_pos = self.np_random.randint(0, 5)
        for i in range(5):
            current_pos = (start_pos + i) % 5
            while player.current_tokens[current_pos] - discard_tokens[current_pos] > best_card_cost[
                current_pos] and num_discard_tokens > 0:
                discard_tokens[current_pos] += 1
                num_discard_tokens -= 1
            if num_discard_tokens == 0:
                break

        # 第二遍
        start_pos = self.np_random.randint(0, 5)
        for i in range(5):
            current_pos = (start_pos + i) % 5
            while player.current_tokens[current_pos] - discard_tokens[current_pos] > 0 and num_discard_tokens > 0:
                discard_tokens[current_pos] += 1
                num_discard_tokens -= 1
            if num_discard_tokens == 0:
                break

        # 第三遍
        discard_tokens.append(0)
        while num_discard_tokens > 0:
            discard_tokens[-1] += 1
            num_discard_tokens -= 1

        return discard_tokens

    def get_legal_actions(self, player):
        """
            根据现有的牌局判断可选动作
        """
        legal_actions = []
        available_num_token = [[], [], []]

        for i in range(42):  # 遍历所有动作
            if i < 5:  # 【第一类动作】 拿同类宝石
                if self.dealer.tokens[i] >= 4:
                    legal_actions.append(i)
            elif i < 15:  # 【第一类动作】 拿不同类宝石
                diff_tokens = FETCH_DIFF_TOKENS[i - 5]  # token索引
                count = 0
                for token in diff_tokens:
                    if self.dealer.tokens[token] > 0:  # 检查有多少可行
                        count += 1
                if count > 0:
                    available_num_token[count-1].append(i)  # 加入当前action

                if i == 14:  # 最后一步进行判断
                    if len(available_num_token[2]) > 0:
                        legal_actions.extend(available_num_token[2])
                    elif len(available_num_token[1]) > 0:
                        legal_actions.extend(available_num_token[1])
                    elif len(available_num_token[0]) > 0:
                        legal_actions.extend(available_num_token[0])
            elif i < 27:  # 【第二类动作】 买牌桌上的牌
                level = (i - 15) // 4 + 1
                pos = (i - 15) % 4
                card = self.dealer.deck[level][pos]

                if card.card_type == -1:  # 跳过
                    continue

                card_cost = [max(card.cost[i] - player.current_card_tokens[i], 0) for i in range(5)]  # 卡的cost
                num_universal_token = player.current_tokens[-1]  # 金色宝石的数目

                flag_buy_deck_cards = 1
                for item in range(5):
                    if card_cost[item] > player.current_tokens[item] + num_universal_token:
                        flag_buy_deck_cards = 0
                        break
                    num_universal_token -= max(card_cost[item] - player.current_tokens[item], 0)  # 更新万能token数目

                if flag_buy_deck_cards:  # 能买
                    legal_actions.append(i)
            elif i < 30:  # 【第二类动作】 买玩家盖住的牌
                pos = i - 27
                if pos < len(player.current_flip_cards):
                    card = player.current_flip_cards[pos]
                    card_cost = [max(card.cost[i] - player.current_card_tokens[i], 0) for i in range(5)]  # 卡的cost
                    num_universal_token = player.current_tokens[-1]  # 金色宝石的数目

                    flag_buy_deck_cards = 1
                    for item in range(5):
                        if card_cost[item] > player.current_tokens[item] + num_universal_token:
                            flag_buy_deck_cards = 0
                            break
                        num_universal_token -= max(card_cost[item] - player.current_tokens[item], 0)  # 更新万能token数目

                    if flag_buy_deck_cards:  # 能买
                        legal_actions.append(i)
            elif i < 42:  # 【第三类动作】 盖牌
                if len(player.current_flip_cards) < 3 and self.dealer.tokens[-1] > 0:  # 还没盖满且有金色卡牌
                    legal_actions.append(i)

        return legal_actions

