from rainbowarena.games.century import Dealer

def generate_action_space(num_upgrade):
    """
    可升级的动作空间
    num_upgrade遍历顺序：0，1，2，3
    可升级的香料：黄（x1），红（x2），绿（x3）（褐色升级不了）
    """
    action_space = []
    for n_upgrade in range(num_upgrade + 1):  # [0, 1, 2, 3]
        for x_1 in range(n_upgrade + 1):
            for x_2 in range(n_upgrade + 1):
                for x_3 in range(n_upgrade + 1):
                    if x_1 + x_2 + x_3 == n_upgrade:
                        action_space.append([x_1, x_2, x_3])
    return action_space


UPGRADE_ACTION_SPACE = generate_action_space(3)  # 最大升级数为3


class CenturyRound:

    def __init__(self, np_random, num_players, Num_upgrade_card, Num_trade_card, Num_token_card, trade_num,
                 players):
        # 【随机库】
        self.np_random = np_random
        # 【玩家信息】
        self.num_players = num_players
        self.current_player = self.np_random.randint(0, self.num_players)  # 随机初始化玩家编号
        # 【dealer信息】
        self.dealer = Dealer(players, self.current_player, self.np_random, num_players)  # 初始化dealer
        # 【各种功能卡的数目】
        self.Num_upgrade_card = Num_upgrade_card
        self.Num_trade_card = Num_trade_card
        self.Num_token_card = Num_token_card
        self.trade_num = trade_num
        self.max_token_num = 10

    def proceed_round(self, player, action):
        """
        动作类型：出牌，购买，休息，换领
        三类功能卡：升级卡，交易卡，香料卡
        ① 出牌
        ①-1 升级卡（Num_upgrade_card）（20，最高的升级卡为3）
        ①-2 交易卡（Num_trade_card）（3，假设最高交易登记为3）
        ①-3 香料卡（Num_token_card）（1）
        ② 购买（6，共6张可购买）
        ③ 休息（1）
        ④ 换领（5，共5张可换领）
        """

        # 【初始化信息】
        Num_upgrade_card = self.Num_upgrade_card
        Num_trade_card = self.Num_trade_card
        Num_token_card = self.Num_token_card
        trade_num = self.trade_num

        # 【①-1 升级卡】
        if action < Num_upgrade_card * 20:
            """
            （1）确定升级数目和升级类型
            （2）执行升级
            （3）更新玩家信息
            """
            # （1）确定升级数目和升级类型
            card_index = action // 20  # 升级2或者升级3，但升级2可能对应多张卡
            if card_index == 0:  # 【升级2】
                for card_idx in range(5):  # 找玩家手牌中第一张升级2
                    if player.map_function_cards[card_idx] != -1:
                        card_index = card_idx
                        break
            else:  # 【升级3】
                card_index = 5
            map_card_index = player.map_function_cards[card_index]  # 全局index -> 玩家手牌index
            card_selected = player.function_cards[map_card_index]  # 玩家手牌index -> 玩家手牌
            card_upgrade = action % 20  # 选择哪种升级方式

            # （2）执行升级
            for token_idx in range(3):
                player.tokens[token_idx] -= UPGRADE_ACTION_SPACE[card_upgrade][token_idx]
                player.tokens[token_idx + 1] += UPGRADE_ACTION_SPACE[card_upgrade][token_idx]

            # （3）更新玩家信息
            player.function_cards.remove(card_selected)  # （1）从手牌中删除
            player.played_function_cards.append(card_selected)  # （2）加入到打出的功能卡堆
            # （3）更新 全局index -> 手牌index
            update_idx_start = map_card_index  # 更新index开始
            update_idx_end = len(player.function_cards)  # 更新index结束
            for function_card_idx in range(update_idx_start, update_idx_end):  # 更新索引
                update_idx = player.function_cards[function_card_idx].card_id  # 全局index
                player.map_function_cards[update_idx] = function_card_idx  # 全局index -> 手牌index（更新）
            player.map_function_cards[card_index] = -1  # 删除所删除牌的手牌index
        # 【①-2 交易卡】
        elif action < Num_upgrade_card * 20 + Num_trade_card * trade_num:
            """
            （1）确定交易类型和交易数目
            （2）执行交易
            （3）更新玩家信息
            """
            # （1）确定交易类型和交易数目
            card_index = (action - Num_upgrade_card * 20) // trade_num + Num_upgrade_card + 4  # 选择第几张卡
            map_card_index = player.map_function_cards[card_index]  # 全局index -> 玩家手牌index
            card_selected = player.function_cards[map_card_index]  # 玩家手牌index -> 玩家手牌
            card_trade = (action - Num_upgrade_card * 20) % trade_num + 1  # 选择哪种交易方式

            # （2）执行交易
            for token_idx in range(4):
                player.tokens[token_idx] -= card_selected.bonus[token_idx] * card_trade
                player.tokens[token_idx] += card_selected.bonus[token_idx + 4] * card_trade

            # （3）更新玩家信息
            player.function_cards.remove(card_selected)  # （1）从手牌中删除
            player.played_function_cards.append(card_selected)  # （2）加入到打出的功能卡堆
            # （3）更新 全局index -> 手牌index
            update_idx_start = map_card_index  # 更新index开始
            update_idx_end = len(player.function_cards)  # 更新index结束
            for function_card_idx in range(update_idx_start, update_idx_end):  # 更新索引
                update_idx = player.function_cards[function_card_idx].card_id  # 全局index
                player.map_function_cards[update_idx] = function_card_idx  # 全局index -> 手牌index（更新）
            player.map_function_cards[card_index] = -1  # 删除所删除牌的手牌index
        # 【①-3 香料卡】
        elif action < Num_upgrade_card * 20 + Num_trade_card * trade_num + Num_token_card:
            """
            （1）确定香料卡类型
            （2）拿香料
            （3）更新玩家信息
            """
            # （1）确定香料卡类型
            card_index = (action - Num_upgrade_card * 20 - Num_trade_card * trade_num) + \
                         Num_upgrade_card + Num_trade_card + 4  # 选择第几张卡
            map_card_index = player.map_function_cards[card_index]  # 全局index -> 玩家手牌index
            card_selected = player.function_cards[map_card_index]  # 玩家手牌index -> 玩家手牌

            # （2）拿香料
            for token_idx in range(4):
                player.tokens[token_idx] += card_selected.bonus[token_idx]

            # （3）更新玩家信息
            player.function_cards.remove(card_selected)  # （1）从手牌中删除
            player.played_function_cards.append(card_selected)  # （2）加入到打出的功能卡堆
            # （3）更新 全局index -> 手牌index
            update_idx_start = map_card_index  # 更新index开始
            update_idx_end = len(player.function_cards)  # 更新index结束
            for function_card_idx in range(update_idx_start, update_idx_end):  # 更新索引
                update_idx = player.function_cards[function_card_idx].card_id  # 全局index
                player.map_function_cards[update_idx] = function_card_idx  # 全局index -> 手牌index（更新）
            player.map_function_cards[card_index] = -1  # 删除所删除牌的手牌index
        # 【② 购买】
        elif action < Num_upgrade_card * 20 + Num_trade_card * trade_num + Num_token_card + 6:
            """
            （1）获取卡信息
            （2）买卡
            """
            # （1）获取卡信息
            card_index = action - Num_upgrade_card * 20 - Num_trade_card * trade_num - Num_token_card  # 选择第几张卡
            card_selected = self.dealer.deck[0][card_index]  # deck index -> 功能卡

            # （2）买卡
            for idx in range(card_index):  # 领功能卡需要付出对应的代价
                for token_idx in range(4):
                    if player.tokens[token_idx] > 0:
                        self.dealer.deck[1][idx][token_idx] += 1
                        player.tokens[token_idx] -= 1
                        break
            for token_idx in range(4):
                player.tokens[token_idx] += self.dealer.deck[1][card_index][token_idx]
                self.dealer.deck[1][card_index][token_idx] = 0
            player.function_cards.append(card_selected)  # 加入手牌
            player.map_function_cards[card_selected.card_id] = len(player.function_cards) - 1  # 存储
            self.dealer.deal_cards(0, card_index)  # 发牌
        # 【③ 休息】
        elif action < Num_upgrade_card * 20 + Num_trade_card * trade_num + Num_token_card + 6 + 1:
            """
            把所有牌重新加入手牌
            """
            # 把所有牌重新加入手牌
            for card_selected in player.played_function_cards:
                player.function_cards.append(card_selected)
                player.map_function_cards[card_selected.card_id] = len(player.function_cards) - 1
            player.played_function_cards.clear()  # 清空
        # 【④ 换领】
        else:
            """
            （1）获取卡信息
            （2）买卡
            """
            # （1）获取卡信息
            card_index = action - Num_upgrade_card * 20 - Num_trade_card * trade_num - Num_token_card - 6 - 1
            card_selected = self.dealer.deck[2][card_index]
            # （2）买卡
            for token_idx in range(4):
                player.tokens[token_idx] -= card_selected.cost[token_idx]
            player.point_cards.append(card_selected)  # 加入牌堆
            player.point += self.dealer.deck[3][card_index]  # 增加金银币对应的积分
            player.point += card_selected.point
            self.dealer.deal_cards(2, card_index)
        # 需要去除多余的token
        token_idx = 0
        while sum(player.tokens) > self.max_token_num:
            if player.tokens[token_idx] > 0:
                player.tokens[token_idx] -= 1
            else:
                token_idx += 1

    def get_legal_actions(self, player):
        """
        动作类型：出牌，购买，休息，换领
        三类功能卡：升级卡，交易卡，香料卡
        ① 出牌
        ①-1 升级卡（Num_upgrade_card）（20，最高的升级卡为3）
        ①-2 交易卡（Num_trade_card）（3，假设最高交易登记为3）
        ①-3 香料卡（Num_token_card）（1）
        ② 购买（6，共6张可购买）
        ③ 休息（1）
        ④ 换领（5，共5张可换领）
        """
        # 【初始化信息】
        legal_actions = []  # 可用动作列表
        Num_upgrade_card = self.Num_upgrade_card
        Num_trade_card = self.Num_trade_card
        Num_token_card = self.Num_token_card
        trade_num = self.trade_num
        action_space = Num_upgrade_card * 20 + Num_trade_card * trade_num + Num_token_card + 6 + 1 + 5  # 动作空间

        for action in range(action_space):
            # 【①-1 升级卡】
            if action < Num_upgrade_card * 20:
                """
                （1）升级卡类型
                （2）判断
                """
                # （1）升级卡类型
                card_index = action // 20
                # （2）判断
                if card_index == 0:
                    is_valid = False
                    # 手牌中有这张卡
                    for card_idx in range(5):
                        if player.map_function_cards[card_idx] != -1:
                            is_valid = True
                            break
                    card_upgrade = action % 20
                    # 升级次数正确，且token足够
                    if card_upgrade >= 10:
                        is_valid = False
                    else:
                        for token_idx in range(3):
                            if player.tokens[token_idx] < UPGRADE_ACTION_SPACE[card_upgrade][token_idx]:
                                is_valid = False
                                break

                    if is_valid is True:
                        legal_actions.append(action)
                else:
                    # 手牌中有这张卡
                    if player.map_function_cards[5] != -1:
                        is_valid = True
                        card_upgrade = action % 20  # 选择哪种升级方式
                        # token足够
                        for token_idx in range(3):
                            if player.tokens[token_idx] < UPGRADE_ACTION_SPACE[card_upgrade][token_idx]:
                                is_valid = False
                                break

                        if is_valid is True:
                            legal_actions.append(action)
            # 【①-2 交易卡】
            elif action < Num_upgrade_card * 20 + Num_trade_card * trade_num:
                """
                （1）升级卡类型
                （2）判断
                """
                # （1）升级卡类型
                card_index = (action - Num_upgrade_card * 20) // trade_num + Num_upgrade_card + 4  # 选择第几张卡
                # （2）判断
                if player.map_function_cards[card_index] != -1:  # 手牌里有这张
                    map_card_index = player.map_function_cards[card_index]  # 全局index -> 玩家手牌index
                    card_selected = player.function_cards[map_card_index]  # 玩家手牌index -> 玩家手牌
                    card_trade = (action - Num_upgrade_card * 20) % trade_num + 1  # 选择哪种交易方式，交易几次

                    is_valid = True

                    # 如果交易之后的token总数大于self.max_token_num
                    if sum(player.tokens) + \
                            card_trade * (sum(card_selected.bonus[4:]) - sum(card_selected.bonus[:4])) > self.max_token_num:
                        is_valid = False

                    # token数目够
                    for token_idx in range(4):
                        if player.tokens[token_idx] < card_selected.bonus[token_idx] * card_trade:
                            is_valid = False
                            break

                    if is_valid is True:
                        legal_actions.append(action)
            # 【①-3 香料卡】
            elif action < Num_upgrade_card * 20 + Num_trade_card * trade_num + Num_token_card:
                """
                （1）香料卡类型
                （2）判断
                """
                # （1）香料卡类型
                card_index = (action - Num_upgrade_card * 20 - Num_trade_card * trade_num) \
                             + Num_upgrade_card + Num_trade_card + 4
                # （2）判断
                if player.map_function_cards[card_index] != -1:  # 手牌里有
                    map_card_index = player.map_function_cards[card_index]  # 全局index -> 玩家手牌index
                    card_selected = player.function_cards[map_card_index]  # 玩家手牌index -> 玩家手牌
                    if sum(player.tokens) + sum(card_selected.bonus) <= self.max_token_num:  # 总数不大于self.max_token_num
                        legal_actions.append(action)
            # 【② 购买】
            elif action < Num_upgrade_card * 20 + Num_trade_card * trade_num + Num_token_card + 6:
                """
                （1）卡类型
                （2）判断
                """
                # （1）卡类型
                card_index = action - Num_upgrade_card * 20 - Num_trade_card * trade_num - Num_token_card  # 选择第几张卡
                # （2）判断
                if self.dealer.deck[0][card_index] != -1 and sum(player.tokens) + \
                        sum(self.dealer.deck[1][card_index]) >= card_index:  # 能有钱购买
                    if sum(player.tokens) + sum(self.dealer.deck[1][card_index]) - card_index <= self.max_token_num:
                        legal_actions.append(action)
            # 【③ 休息】
            elif action < Num_upgrade_card * 20 + Num_trade_card * trade_num + Num_token_card + 6 + 1:
                legal_actions.append(action)
            # 【④ 换领】
            else:
                """
                （1）卡类型
                （2）判断
                """
                # （1）卡类型
                card_index = action - Num_upgrade_card * 20 - Num_trade_card * trade_num - Num_token_card - 6 - 1
                card_selected = self.dealer.deck[2][card_index]
                # （2）判断
                is_valid = True
                for token_idx in range(4):
                    if player.tokens[token_idx] < card_selected.cost[token_idx]:
                        is_valid = False
                        break
                if is_valid is True:
                    legal_actions.append(action)

        return legal_actions
