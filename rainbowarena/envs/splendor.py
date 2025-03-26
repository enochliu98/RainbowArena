import numpy as np
from collections import OrderedDict

from rainbowarena.envs import Env
from rainbowarena.games.splendor import Game
from rainbowarena.games.splendor.player import SplendorPlayer
from rainbowarena.games.splendor.dealer import SplendorDealer
from rainbowarena.games.splendor.card import SplendorCard
import torch
import copy

FETCH_DIFF_TOKENS = [[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3],
                     [0, 3, 4], [0, 2, 4], [0, 2, 3],
                     [0, 1, 4], [0, 1, 3],
                     [0, 1, 2]]

class SplendorEnv(Env):

    def __init__(self, config):
        self.name = 'splendor'
        self.game = Game(2)
        super().__init__(config)

        # 发展卡对应宝石（5）+ 宝石（6）+ 盖住的牌（7*3）+ 分数（1）
        self.state_player_len = 5 + 6 + 7 * 3 + 1 
        # 宝石（6） + 发展卡（7*12）+ 贵族卡（7 * (self.num_players+1)）
        self.state_deck_len = 6 + 7 * 12 + 7 * (self.num_players+1)
        # 玩家数目（self.num_players）
        self.state_unique_len = 7
        # 历史行动
        self.state_history_action_len = 10 * 5 * 2  
        # 上述三者相加
        self.state_len = self.state_player_len * 2 + self.state_deck_len + self.state_unique_len + self.state_history_action_len

        self.state_shape = [[self.state_len] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def _extract_state(self, state):
        obs = np.zeros((self.state_len), dtype=int)
        obs[: self.state_player_len * 2] = extract_player_state(state, self.num_players)
        obs[self.state_player_len * 2 : self.state_player_len * 2 + self.state_deck_len] = extract_deck_state(state, self.num_players)
        obs[self.state_player_len * 2 + self.state_deck_len: self.state_player_len * 2 + self.state_deck_len + self.state_unique_len] = extract_unique_state_embedding(state, self.num_players)
        obs[self.state_player_len * 2 + self.state_deck_len + self.state_unique_len:] = extract_history_action_state(state)

        legal_action_id = self._get_legal_actions()
        extracted_state = {'obs': obs, 'legal_actions': legal_action_id}
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = self.game.get_legal_actions()
        extracted_state['action_record'] = self.action_recorder  
        return extracted_state

    ##########【For MCTS】###########

    def reset_simulate(self):
        '''
            ouput:状态,隐藏信息,玩家
        '''
        state, player_id = self.game.init_game()
        info = [state['invisible_info']]
        info.append(state["self_cards"])
        info.append(state["other_cards"])
        info.append(state["self_noble_cards"])
        info.append(state["other_noble_cards"])
        
        return self._extract_state(state)['obs'], info, player_id

    def step_simulate(self, state, info, player_id, action):
        '''
            input: 状态,隐藏信息,玩家,动作
            output: 状态,隐藏信息,玩家
        '''
        # 复制原始信息
        state_tmp = copy.deepcopy(state)
        info_tmp = copy.deepcopy(info)

        new_state = np.zeros((self.state_len), dtype=int)

        # 【新我方信息：敌方变我方】 这部分信息不变
        new_state[: self.state_player_len] = state_tmp[self.state_player_len: self.state_player_len*2]  
        # 【新独有信息】 这部分信息不变
        new_state[self.state_player_len * 2 + self.state_deck_len: self.state_player_len * 2 + 
                  self.state_deck_len + self.state_unique_len] = extract_unique_state_simulate(1-player_id)
       
        # 根据执行动作玩家的状态，构造对应的player和dealer
        player = SplendorPlayer()
        player.current_tokens = state_tmp[:6]  # tokens
        player.current_card_tokens = state_tmp[6:11]  # card_tokens
        player.current_flip_cards = deextract_card_state(state_tmp[11:32])  # filp_cards 这里需特殊处理
        player.point = state_tmp[32]  # point
        player.current_cards = info_tmp[1] 
        player.current_noble_cards = info_tmp[3]
        dealer = SplendorDealer()
        dealer.tokens = state_tmp[66:72]  # tokens
        dealer.cards = info_tmp[0]  # cards
        dealer.deck = [deextract_card_state(state_tmp[72:72+7*3], 0), 
                       deextract_card_state(state_tmp[72+7*3:72+7*7], 1), 
                       deextract_card_state(state_tmp[72+7*7:72+7*11], 2), 
                       deextract_card_state(state_tmp[72+7*11:72+7*15], 3)]
        # 【新历史动作信息】 这部分信息不变
        new_state[self.state_player_len * 2 + self.state_deck_len + self.state_unique_len:] = extract_history_action_state_simulate(
            state_tmp[self.state_player_len * 2 + self.state_deck_len + self.state_unique_len:], action, player, dealer) 
        # 【新敌方信息,新牌桌信息】
        player, dealer = self.proceed_round(player, dealer, action)  # 改变牌局，新的玩家信息和新的牌局信息
        new_state[self.state_player_len: self.state_player_len * 2] = extract_player_state_simulate(player)
        new_state[self.state_player_len * 2 : self.state_player_len * 2 + self.state_deck_len] = extract_deck_state_simulate(dealer)


        new_info = [dealer.cards]
        new_info.append(info_tmp[2])
        new_info.append(player.current_cards)
        new_info.append(info_tmp[4])
        new_info.append(player.current_noble_cards)

        return new_state, new_info, 1-player_id

    def is_over_simulate(self, state, info, player_id):
        
        state_tmp = copy.deepcopy(state)
        info_tmp = copy.deepcopy(info)

        player_self = SplendorPlayer()
        player_self.current_tokens = state_tmp[:6]  # tokens
        player_self.current_card_tokens = state_tmp[6:11]  # card_tokens
        player_self.current_flip_cards = deextract_card_state(state_tmp[11:32])  # filp_cards 这里需特殊处理
        player_self.point = state_tmp[32]  # point
        player_self.current_cards = info_tmp[1] 
        player_self.current_noble_cards = info_tmp[3]

        player_other = SplendorPlayer()
        player_other.current_tokens = state_tmp[self.state_player_len:self.state_player_len+6]
        player_other.current_card_tokens = state_tmp[self.state_player_len+6:self.state_player_len+11]
        player_other.current_flip_cards = deextract_card_state(state_tmp[self.state_player_len+11:self.state_player_len+32])
        player_other.point = state_tmp[self.state_player_len+32]
        player_other.current_cards = info_tmp[2]
        player_other.current_noble_cards = info_tmp[4]

        if player_id == 0:
            players = [player_self, player_other]
        else:
            players = [player_other, player_self]

        round_over_player = self.game.round_over_player
        
        winner_id = -1  # 初始化

        if player_id == round_over_player: # 当前回合结束
            winner_id = self.game.judger.judge_winner(players, round_over_player)
        

        return winner_id

    def get_validmoves_simulate(self, state, info):
        state_tmp = copy.deepcopy(state)
        info_tmp = copy.deepcopy(info)

        player = SplendorPlayer()
        player.current_tokens = state_tmp[:6]  # tokens
        player.current_card_tokens = state_tmp[6:11]  # card_tokens
        player.current_flip_cards = deextract_card_state(state_tmp[11:32])  # filp_cards 这里需特殊处理
        player.point = state_tmp[32]  # point
        player.current_cards = info_tmp[1] 
        player.current_noble_cards = info_tmp[3]
        dealer = SplendorDealer()
        dealer.tokens = state_tmp[66:72]  # tokens
        dealer.cards = info_tmp[0]  # cards
        dealer.deck = [deextract_card_state(state_tmp[72:72+7*3], 0), 
                       deextract_card_state(state_tmp[72+7*3:72+7*7], 1), 
                       deextract_card_state(state_tmp[72+7*7:72+7*11], 2), 
                       deextract_card_state(state_tmp[72+7*11:72+7*15], 3)]

        legal_actions = self.get_legal_actions(player, dealer)
        valid_moves = [0] * 42
        for idx in range(42):
            if idx in legal_actions:
                valid_moves[idx] = 1
    
        return valid_moves

    def proceed_round(self, player, dealer, action):
        # action (5+10)+12+12=39
        if action < 5: # 第一类动作，拿相同颜色宝石 【0，4】
            player.current_tokens[action] += 2 
            dealer.tokens[action] -= 2  
            if sum(player.current_tokens) > 10:
                discard_tokens = self.get_discard_tokens(player, dealer)
                for i in range(6):
                    player.current_tokens[i] -= discard_tokens[i]
        elif action < 15: # 第一类动作，拿不同颜色的宝石 【5,14】
            if dealer.tokens[FETCH_DIFF_TOKENS[action - 5][0]] > 0:  # 必然的
                player.current_tokens[FETCH_DIFF_TOKENS[action - 5][0]] += 1
                dealer.tokens[FETCH_DIFF_TOKENS[action - 5][0]] -= 1
            if dealer.tokens[FETCH_DIFF_TOKENS[action - 5][1]] > 0:
                player.current_tokens[FETCH_DIFF_TOKENS[action - 5][1]] += 1
                dealer.tokens[FETCH_DIFF_TOKENS[action - 5][1]] -= 1
            if dealer.tokens[FETCH_DIFF_TOKENS[action - 5][2]] > 0:
                player.current_tokens[FETCH_DIFF_TOKENS[action - 5][2]] += 1
                dealer.tokens[FETCH_DIFF_TOKENS[action - 5][2]] -= 1

            if sum(player.current_tokens) > 10:
                discard_tokens = self.get_discard_tokens(player, dealer)
                for i in range(6):
                    player.current_tokens[i] -= discard_tokens[i]
                    dealer.tokens[i] += discard_tokens[i]
        elif action < 27: # 第二类动作，买牌 【15,26】
            # 计算牌的位置
            level = (action - 15) // 4 + 1
            pos = (action - 15) % 4

            card = dealer.deck[level][pos]  # 获取牌桌上的牌
            for i, token_cost in enumerate(card.cost):  
                token_cost_minus_current_card_tokens = max(token_cost - player.current_card_tokens[i], 0)
                player_cost_tokens_1 = min(player.current_tokens[i], token_cost_minus_current_card_tokens)
                player_cost_tokens_2 = max(token_cost_minus_current_card_tokens - player.current_tokens[i], 0)

                player.current_tokens[i] -= player_cost_tokens_1 # 去掉玩家宝石，当前卡的cost减去bonus宝石
                dealer.tokens[i] += player_cost_tokens_1  # 增加牌桌上的宝石
                player.current_tokens[-1] -= player_cost_tokens_2  # 去掉玩家万能宝石
                dealer.tokens[-1] += player_cost_tokens_2  # 增加牌桌上的万能宝石

            dealer.deal_cards(level, pos)  # 发牌补充
            player.current_cards.append(card)  # 更新玩家手牌
            player.point += card.point  # 更新玩家积分
            player.current_card_tokens[card.bonus] += 1  # 更新宝石矿卡对应宝石

        elif action < 30: # 第二类动作，买盖着的牌 【27，29】
            pos = action - 27

            card = player.current_flip_cards[pos]  # 获取玩家盖着的牌

            for i, token_cost in enumerate(card.cost):  
                token_cost_minus_current_card_tokens = max(token_cost - player.current_card_tokens[i], 0)
                player_cost_tokens_1 = min(player.current_tokens[i], token_cost_minus_current_card_tokens)
                player_cost_tokens_2 = max(token_cost_minus_current_card_tokens - player.current_tokens[i], 0)

                player.current_tokens[i] -= player_cost_tokens_1 # 去掉玩家宝石，当前卡的cost减去bonus宝石
                dealer.tokens[i] += player_cost_tokens_1  # 增加牌桌上的宝石
                player.current_tokens[-1] -= player_cost_tokens_2  # 去掉玩家万能宝石
                dealer.tokens[-1] += player_cost_tokens_2  # 增加牌桌上的万能宝石
                
            player.current_flip_cards.pop(pos)  # 去掉
            player.current_cards.append(card)  # 更新玩家手牌
            player.point += card.point  # 更新玩家积分
            player.current_card_tokens[card.bonus] += 1  # 更新宝石矿卡对应宝石

        elif action < 42: # 第三类动作，盖牌 【30,41】
            # 计算牌的位置
            level = (action - 30) // 4 +1
            pos = (action - 30) % 4

            card = dealer.deck[level][pos]  # 获取牌桌上的牌
            dealer.deal_cards(level, pos)  # 发牌补充
            player.current_flip_cards.append(card)  # 盖上对应的牌

            player.current_tokens[-1] += 1
            dealer.tokens[-1] -= 1


        for noble_card in dealer.deck[0]:
            flag = 1  # 初始化标志,1代表匹配成功，0代表匹配失败
            for i in range(5):
                if player.current_card_tokens[i] < noble_card.cost[i]:
                    flag = 0  # 设置标志
                    break
            if flag:
                player.current_noble_cards.append(noble_card)  # 更新玩家贵族卡
                dealer.deck[0].remove(noble_card)  # 删除牌桌上的贵族卡
                player.point += noble_card.point  # 更新玩家积分
                break  # 只要找到一张就退出


        return player, dealer

    def get_discard_tokens(self, player, dealer):
        """
            寻找要删掉的牌
        """
        best_card = None
        min_dis = 1000
        # 先检查牌面上的牌
        for level in range(1, 4):
            for pos in range(4):
                dis = 0  # 初始化距离
                card = dealer.deck[level][pos]  # 当前卡
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
            while player.current_tokens[current_pos] - discard_tokens[current_pos] > best_card_cost[current_pos] and num_discard_tokens > 0:
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

    def get_legal_actions(self, player, dealer):
        """
            根据现有的牌局判断可选动作
        """
        legal_actions = []
        available_num_token = [[], [], []]
        
        for i in range(42): # 遍历所有动作
            if i < 5: # 【第一类动作】 拿同类宝石
                if dealer.tokens[i] >= 4:
                    legal_actions.append(i)
            elif i < 15: # 【第一类动作】 拿不同类宝石
                diff_tokens = FETCH_DIFF_TOKENS[i - 5]  # token索引
                for num, token in enumerate(diff_tokens):
                    if dealer.tokens[token] > 0:
                        available_num_token[num].append(i)  # 加入当前action
                    else:
                        break 

                if i == 14: # 最后一步进行判断
                    if len(available_num_token[2]) > 0:
                        legal_actions.extend(available_num_token[2])
                    elif len(available_num_token[1]) > 0:
                        legal_actions.extend(available_num_token[1])
                    elif len(available_num_token[0]) > 0:
                        legal_actions.extend(available_num_token[0])
            elif i < 27:  # 【第二类动作】 买牌桌上的牌
                level = (i - 15) // 4 + 1
                pos = (i - 15) % 4

                # print(level, pos, len(dealer.deck[1]), len(dealer.deck[2]), len(dealer.deck[3]))
                card = dealer.deck[level][pos]

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
            elif i < 30: # 【第二类动作】 买玩家盖住的牌
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
                        num_universal_token -= max(card_cost[item] - player.current_tokens[item], 0) # 更新万能token数目
                    
                    if flag_buy_deck_cards:  # 能买
                        legal_actions.append(i)
            elif i < 42: # 【第三类动作】 盖牌
                if len(player.current_flip_cards) < 3 and dealer.tokens[-1] > 0: # 还没盖满且有金色卡牌
                    legal_actions.append(i)
        
        return legal_actions

    ##########【For MCTS】########### 

    def get_payoffs(self):
        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id):
        return action_id # 统一的动作编码

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()  # 获取可用动作
        legal_ids = {action: None for action in legal_actions}  # 转化成dict
        return OrderedDict(legal_ids)


def extract_player_state(state, num_players):
    player_state = []

    # 我方的信息
    player_state.extend(state["self_tokens"])
    player_state.extend(state["self_card_tokens"])
    player_state.extend(extract_card_state(state["self_flip_cards"], 3))
    player_state.append(state["self_point"])

    # 对方的信息
    player_state.extend(state["other_tokens"])
    player_state.extend(state["other_card_tokens"])
    player_state.extend(extract_card_state(state["other_flip_cards"], 3))
    player_state.append(state["other_point"])

    return np.array(player_state)

def extract_deck_state(state, num_players):
    deck_state = []
    
    deck_state.extend(state["deck_tokens"])
    deck_state.extend(extract_card_state(state["deck_cards"][0], num_players+1))
    deck_state.extend(extract_card_state(state["deck_cards"][1], 4))
    deck_state.extend(extract_card_state(state["deck_cards"][2], 4))
    deck_state.extend(extract_card_state(state["deck_cards"][3], 4))
    
    return np.array(deck_state)

def extract_unique_state(state, num_players):
    '''
    one hot编码
    '''
    
    unique_state = [0] * num_players

    unique_state[state["self_player"]] = 1

    return np.array(unique_state)

def extract_unique_state_embedding(state, num_players):
    '''
    embedding编码
    '''
    embedding = torch.nn.Embedding(num_players, 7)
    player = torch.tensor([state["self_player"]])
    embedding_output = embedding(player)[0].detach().numpy()

    return embedding_output

def extract_card_state(cards, max_len):
    card_state = []
    num = 0
    for card in cards:  # 遍历所有的卡
        card_state.extend(card.cost)
        card_state.append(card.point)
        card_state.append(card.bonus)
        num += 1  # 保存的卡的数目增加1
        if num == max_len:  # 如果卡数目达标
            break 

    while num < max_len:
        card_state.extend([-1] * 7)  # 添加默认卡
        num += 1 

    return card_state

def extract_history_action_state(state):
    history_action = state["history_action"]
    self_history_action = []
    other_history_action = []
    self_history_action_state = []
    other_history_action_state = []

    for action in history_action:
        if action[0] == state["self_player"]:
            self_history_action.append(action[1])
        else:
            other_history_action.append(action[1])
    
    count = 0
    for action_state in self_history_action[::-1]:
        self_history_action_state.extend(action_state)
        count += 1
        if count == 5:
            break
    while count < 5:
        self_history_action_state.extend([-1] * 10)
        count += 1

    count = 0
    for action_state in other_history_action[::-1]:
        other_history_action_state.extend(action_state)
        count += 1
        if count == 5:
            break
    while count < 5:
        other_history_action_state.extend([-1] * 10)
        count += 1

    history_action_state = self_history_action_state + other_history_action_state

    return history_action_state

def extract_history_action_state_baseline(state):
    history_action = state["history_action_baseline"]
    self_history_action = []
    other_history_action = []
    self_history_action_state = []
    other_history_action_state = []

    for action in history_action:
        if action[0] == state["self_player"]:
            self_history_action.append(action[1])
        else:
            other_history_action.append(action[1])
    
    count = 0
    for action_state in self_history_action[::-1]:
        self_history_action_state.extend(action_state)
        count += 1
        if count == 5:
            break
    while count < 5:
        self_history_action_state.extend([-1] * 2)
        count += 1

    count = 0
    for action_state in other_history_action[::-1]:
        other_history_action_state.extend(action_state)
        count += 1
        if count == 5:
            break
    while count < 5:
        other_history_action_state.extend([-1] * 2)
        count += 1

    history_action_state = self_history_action_state + other_history_action_state

    return history_action_state

def deextract_card_state(state, level=-1):
    '''
    将状态向量转化为牌的信息
    '''
    len_cards = len(state) // 7
    cards = []
    for i in range(len_cards):
        info = state[i * 7: (i + 1) * 7] 
        if sum(info) > 0:
            card = SplendorCard(card_type=level, cost=info[:-2], point=info[-2], bonus=info[-1])
            cards.append(card)
        elif sum(info) == 0:
            card = SplendorCard()
            cards.append(card)
    return cards

def extract_player_state_simulate(player):
    player_state = []

    # 我方信息
    player_state.extend(player.current_tokens)
    player_state.extend(player.current_card_tokens)
    player_state.extend(extract_card_state(player.current_flip_cards, 3))
    player_state.append(player.point)

    return np.array(player_state)

def extract_deck_state_simulate(dealer):
    deck_state = []

    deck_state.extend(dealer.tokens)
    deck_state.extend(extract_card_state(dealer.deck[0], 3))
    deck_state.extend(extract_card_state(dealer.deck[1], 4))
    deck_state.extend(extract_card_state(dealer.deck[2], 4))
    deck_state.extend(extract_card_state(dealer.deck[3], 4))

    return np.array(deck_state)

def extract_unique_state_simulate(player_id):
    embedding = torch.nn.Embedding(2, 7)
    player = torch.tensor([player_id])
    embedding_output = embedding(player)[0].detach().numpy()

    return embedding_output

def extract_history_action_state_simulate(state_history, action, player, dealer):
    self_history_action_state = list(state_history[50:])
    other_history_action_state = list(state_history[10:50]) # 历史动作信息

    info_card = []
    info_token = [0] * 6
    info_category = []
    info_action = [action]

    if action < 5:
        info_token[action] += 2
        info_card.extend([-1, -1])
        info_category.append(0)
    elif action < 15:
        info_token[FETCH_DIFF_TOKENS[action - 5][0]] += 1
        info_token[FETCH_DIFF_TOKENS[action - 5][1]] += 1
        info_token[FETCH_DIFF_TOKENS[action - 5][2]] += 1
        info_card.extend([-1, -1])
        info_category.append(1)
    elif action < 27:
        level = (action - 15) // 4 + 1
        pos = (action - 15) % 4
        card = dealer.deck[level][pos]  # 获取牌桌上的牌
        for i in range(5):
            info_token[i] = -card.cost[i]
        info_card.append(card.point)
        info_card.append(card.bonus)
        info_category.append(2)
    elif action < 30:
        pos = action - 27
        card = player.current_flip_cards[pos]  # 获取玩家盖着的牌
        for i in range(5):
            info_token[i] = -card.cost[i]
        info_card.append(card.point)
        info_card.append(card.bonus)
        info_category.append(3)
    elif action < 42:
        info_token[-1] += 1
        level = (action - 30) // 4 +1
        pos = (action - 30) % 4
        card = dealer.deck[level][pos]  # 获取牌桌上的牌        
        for i in range(5):
            info_token[i] = -card.cost[i]
        info_card.append(card.point)
        info_card.append(card.bonus)    
        info_category.append(4)

    info = info_card + info_token + info_category + info_action  # 2 + 6 + 1 + 1 = 10
    other_history_action_state.extend(info)

    history_action_state = self_history_action_state + other_history_action_state 

    return np.array(history_action_state)


