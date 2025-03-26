import numpy as np
from collections import OrderedDict

from rainbowarena.envs import Env
from rainbowarena.games.splendor_v2 import Game
import torch

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
