import openai 
import random
from typing import Dict, Optional
from pathlib import Path
import re
import time
# from base_llm_agent import BaseLLMAgent
from rainbowarena.agents.llm_agents.base_llm_agent import BaseLLMAgent


class MahjongAgent(BaseLLMAgent):
    def __init__(
            self,
            llm_info: Optional[Dict[str, str]] = None,
        ):
            self.llm_info = llm_info
            self.stop_token = None
            for key,value in llm_info.items():
                try:
                    value = float(value)
                except ValueError:
                    pass
                setattr(self,key,value)
            self.messages = [{"role": "system", "content": self.prompt_rule}]
            self.act2des = {}
            self.illegal_counts = 0
            self.retry_counts = 0
            self.match_one = 0
            self.match_two = 0
            self.random_counts = 0
            self.use_raw = False

            # Define colors and their corresponding offsets
            colors = {
                "bamboo": 0,
                "characters": 9,
                "dots": 18,
            }
            # Generate number cards and action cards for each color
            for color, offset in colors.items():
                self.act2des.update({
                    **{i + offset: f"{color}-{i+1}" for i in range(10)},
                })
            self.act2des.update({
                27: "dragons-green",
                28: "dragons-red",
                29: "dragons-white",
                30: "winds-east",
                31: "winds-west",
                32: "winds-north",
                33: "winds-south",
                34: "pong",
                35: "chow",
                36: "gong",
                37: "stand",
                                 })
            self.pos2act = {value:key for (key,value) in self.act2des.items()}

    def match(self,answer):
        pattern = r"Action: (.+)"
        match = re.search(pattern, answer)
        return match

    def _get_players_pile(self, observation):
        players_pile = {}
        for key, value in observation['raw_obs']['players_pile'].items():
            players_pile[key] = [[item.get_str() for item in sublist] for sublist in value]
        
        return players_pile

    def _get_legal_actions(self, observation):
        legal_actions = []
        legal_actions_id = observation['legal_actions']
        action_mask = list(legal_actions_id.keys())
        for i in action_mask:
            legal_actions.append(self.act2des[i])
        return legal_actions

    def _get_board_status(self, observation):
        
        self.hand = [i.get_str() for i in observation['raw_obs']['current_hand']]
        self.sorted_hand = sorted(self.hand, key=self.tile_sort_key)
        self.table = [i.get_str() for i in observation['raw_obs']['table']]
        self.sorted_table = sorted(self.table, key=self.tile_sort_key)
        self.players_pile = self._get_players_pile(observation)
        self.legal_actions = self._get_legal_actions(observation)
        self.sorted_legal_actions = sorted(self.legal_actions, key=self.tile_sort_key)

    def tile_sort_key(self, tile):
        # 分离类别和数字/颜色

        # 定义类别的排序优先级
        category_order = {
            'bamboo': 0,
            'characters': 1,
            'dots': 2,
            'dragons': 3,
            'winds': 4,
            'pong': 100,
            'chow': 101,
            'gong':102,
            'stand': 103,
        }

        # 如果是动作，则直接返回其排序优先级
        if tile in category_order:
            return (category_order[tile], float('inf'))
        
        # 否则，继续按麻将牌处理
        parts = tile.split('-')
        category = parts[0]
        value = parts[1]
        
        # 数字化数字部分，如果不是数字（如winds-east），则赋值为一个高值以确保其排序在后
        if value.isdigit():
            value = int(value)
        else:
            value = float('inf')
        
        # 返回 (类别优先级, 数字/颜色)
        return (category_order[category], value)

    def _get_chat_action(self, action_mask, observation, info= None):
        max_retries = 3
        current_retry = 0
        
        while current_retry < max_retries:
            try:
                self._get_board_status(observation=observation)
                message = f"Your observation now is :\n"
                message += f"You are player {observation['raw_obs']['player']}.\n"
                message += f"Your hand now is:\n"
                message += f"{self.sorted_hand}\n"
                message += f"The played cards on the table now is:\n "
                message += f"{self.sorted_table}\n"
                message += f"The public piles of each players are: \n"
                for idx, pile in self.players_pile.items():
                    message += f"player {idx}'s pile: {pile}.\n"
                message += "You should think step by step and output your action. For example: 'bamboo-6'. Please make sure your output is in lowercase.\n"
                message += f"Now you can and only choose one of the following legal actions:\n"
                message += f"{self.sorted_legal_actions} \n"
                message += f"You will respond with an action, formatted as:\n Action: <action>\n where you replace <action> with your actual action.\n"
                message += f"You need to make sure the action you choose is legal. \n"
                message += f"\nYou should explain why you choose the action\n"
                print(message)
                answer = self._get_chat_answer(message=message)
                match = self.match(answer)
                if match:
                    pos = match.group(1)
                    pos = pos.rstrip('.')
                    print(f"\n提取到的行动: {pos} \n")
                    action = self.pos2act.get(pos, -999)
                    # action == None代表匹配动作失败
                    if action == -999:
                        current_retry += 1
                        self.retry_counts += 1
                        self.match_two += 1
                        print("匹配动作空间失败！")
                    # 表示匹配到了但是动作不合法
                    else:
                        if action in action_mask:
                            break
                        else: 
                            print("It is an illegal action!")
                            current_retry += 1
                            self.retry_counts += 1
                            self.illegal_counts += 1
                else:
                    print("匹配 Action:<action> 失败！")
                    current_retry += 1
                    self.retry_counts += 1
                    self.match_one += 1
            except Exception as e:
                print(e.__context__, e)
                current_retry += 1
                self.retry_counts += 1
                time.sleep(1)
        # 如果匹配失败或者匹配到了但是动作不合法，则retry
        # retry3次如果没有动作返回，则返回随机动作。
        else:
            action = random.choice(action_mask)
            self.random_counts += 1
            print("random action:", action)

        return action
    
    def act(self, player_id, observation, info=None):
        
        # action_space = self.env.action_space
        action_space = list(range(37))
        legal_actions = observation['legal_actions']
        action_mask = list(legal_actions.keys())
        try:
            action = self._get_chat_action(action_mask, observation, info=None)
        except Exception as e:
            print(e)
            action = random.choice(action_mask)
            self.random_counts += 1
        return action  