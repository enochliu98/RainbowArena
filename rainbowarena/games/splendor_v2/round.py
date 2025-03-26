from rainbowarena.games.splendor_v2 import Dealer

FETCH_DIFF_TOKENS = [
                     [0], [1], [2], [3], [4],  # 5
                     [0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4],  # 10
                     [0, 3, 4], [0, 2, 4], [0, 2, 3], [0, 1, 4], [0, 1, 3], [0, 1, 2], [2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3],  # 10
                     ]


class SplendorRound:
    ''' Round can call other Classes' functions to keep the game running
    '''

    def __init__(self, np_random, num_players):
        self.np_random = np_random
        self.dealer = Dealer(self.np_random, num_players)  # 初始化dealer
        self.num_players = num_players
        self.current_player = self.np_random.randint(0, self.num_players)  # 随机初始化玩家编号
        self.num_actions = 57

    def proceed_round(self, player, action):
        if action < 5:  # 第一类动作，拿相同颜色宝石 【0，4】
            player.current_tokens[action] += 2
            self.dealer.tokens[action] -= 2
        elif action < 30:  # 第一类动作，拿不同颜色的宝石 【5,29】
            for i in range(len(FETCH_DIFF_TOKENS[action-5])):
                player.current_tokens[FETCH_DIFF_TOKENS[action - 5][i]] += 1
                self.dealer.tokens[FETCH_DIFF_TOKENS[action - 5][i]] -= 1
        elif action < 42:  # 第二类动作，买牌 【15,26】
            # 计算牌的位置
            level = (action - 30) // 4 + 1
            pos = (action - 30) % 4

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
        elif action < 45:  # 第二类动作，买盖着的牌 【27，29】
            pos = action - 42

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
        elif action < 57:  # 第三类动作，盖牌 【30,41】
            # 计算牌的位置
            level = (action - 45) // 4 + 1
            pos = (action - 45) % 4

            card = self.dealer.deck[level][pos]  # 获取牌桌上的牌
            self.dealer.deal_cards(level, pos)  # 发牌补充
            player.current_flip_cards.append(card)  # 盖上对应的牌

            if self.dealer.tokens[-1] > 0 and sum(player.current_tokens) < 10:  # 牌桌上有万能卡
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

    def get_legal_actions(self, player):
        """
            根据现有的牌局判断可选动作
            5+25+12+3+12=57
            ①：5
            ②：25
            ③：12
            ④：3
            ⑤：12
        """
        legal_actions = []

        for i in range(self.num_actions):  # 遍历所有动作
            if i < 5:  # 【第一类动作】 拿同类宝石
                count = min(max(10 - sum(player.current_tokens), 0), 3)  # 还可以拿的宝石数,最少为0,最大为3
                if self.dealer.tokens[i] >= 4 and count >= 2:
                    legal_actions.append(i)
            elif i < 30:  # 【第一类动作】 拿不同类宝石
                count = min(max(10 - sum(player.current_tokens), 0), 3)  # 还可以拿的宝石数,最少为0,最大为3
                if count == 3:
                    diff_tokens = FETCH_DIFF_TOKENS[i - 5]  # token索引
                    flag = 1
                    for token in diff_tokens:
                        if self.dealer.tokens[token] <= 0:  # 没有
                            flag = 0
                            break
                    if flag:
                        legal_actions.append(i)
                elif count == 2:
                    diff_tokens = FETCH_DIFF_TOKENS[i - 5]  # token索引
                    if len(diff_tokens) > 2:
                        continue
                    flag = 1
                    for token in diff_tokens:
                        if self.dealer.tokens[token] <= 0:  # 没有
                            flag = 0
                            break
                    if flag:
                        legal_actions.append(i)
                elif count == 1:
                    diff_tokens = FETCH_DIFF_TOKENS[i - 5]  # token索引
                    if len(diff_tokens) > 1:
                        continue
                    flag = 1
                    for token in diff_tokens:
                        if self.dealer.tokens[token] <= 0:  # 没有
                            flag = 0
                            break
                    if flag:
                        legal_actions.append(i)
            elif i < 42:  # 【第二类动作】 买牌桌上的牌
                level = (i - 30) // 4 + 1
                pos = (i - 30) % 4
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
            elif i < 45:  # 【第二类动作】 买玩家盖住的牌
                pos = i - 42
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
            elif i < 57:  # 【第三类动作】 盖牌
                if len(player.current_flip_cards) < 3:
                    legal_actions.append(i)

        return legal_actions
