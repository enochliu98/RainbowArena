import re
import time
import random
from typing import Dict, Optional
from rainbowarena.agents.llm_agents.base_llm_agent import BaseLLMAgent

card_id2des = {
    0: "1 of Heart",
    1: "1 of Club",
    2: "1 of Spade",
    3: "1 of Diamond",
    4: "2 of Heart",
    5: "2 of Club",
    6: "2 of Spade",
    7: "2 of Diamond",
    8: "3 of Heart",
    9: "3 of Club",
    10: "3 of Spade",
    11: "3 of Diamond",
    12: "4 of Heart",
    13: "4 of Club",
    14: "4 of Spade",
    15: "4 of Diamond",
    16: "5 of Heart",
    17: "5 of Club",
    18: "5 of Spade",
    19: "5 of Diamond",
    20: "6 of Heart",
    21: "6 of Club",
    22: "6 of Spade",
    23: "6 of Diamond",
    24: "7 of Heart",
    25: "7 of Club",
    26: "7 of Spade",
    27: "7 of Diamond",
    28: "8 of Heart",
    29: "8 of Club",
    30: "8 of Spade",
    31: "8 of Diamond",
    32: "9 of Heart",
    33: "9 of Club",
    34: "9 of Spade",
    35: "9 of Diamond",
    36: "10 of Heart",
    37: "10 of Club",
    38: "10 of Spade",
    39: "10 of Diamond",
    40: "Payoo 1",
    41: "Payoo 2",
    42: "Payoo 3",
    43: "Payoo 4",
    44: "Payoo 5",
    45: "Payoo 6",
    46: "Payoo 7",
    47: "Payoo 8",
    48: "Payoo 9",
    49: "Payoo 10",
    50: "Payoo 11",
    51: "Payoo 12",
    52: "Payoo 13",
    53: "Payoo 14",
    54: "Payoo 15",
    55: "Payoo 16",
    56: "Payoo 17",
    57: "Payoo 18",
    58: "Payoo 19",
    59: "Payoo 20",
}

color_id2des = {
    -1: 'No Restriction',
    0: 'Heart',
    1: 'Club',
    2: 'Spade',
    3: 'Diamond',
    4: 'Payoo'
}

class PapayooAgent(BaseLLMAgent):

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
            self.illegal_counts = 0
            self.retry_counts = 0
            self.match_one = 0
            self.match_two = 0
            self.random_counts = 0
            self.use_raw = False
            self.act2des = {
                0: "Playing the 1 of Heart",
                1: "Playing the 1 of Club",
                2: "Playing the 1 of Spade",
                3: "Playing the 1 of Diamond",
                4: "Playing the 2 of Heart",
                5: "Playing the 2 of Club",
                6: "Playing the 2 of Spade",
                7: "Playing the 2 of Diamond",
                8: "Playing the 3 of Heart",
                9: "Playing the 3 of Club",
                10: "Playing the 3 of Spade",
                11: "Playing the 3 of Diamond",
                12: "Playing the 4 of Heart",
                13: "Playing the 4 of Club",
                14: "Playing the 4 of Spade",
                15: "Playing the 4 of Diamond",
                16: "Playing the 5 of Heart",
                17: "Playing the 5 of Club",
                18: "Playing the 5 of Spade",
                19: "Playing the 5 of Diamond",
                20: "Playing the 6 of Heart",
                21: "Playing the 6 of Club",
                22: "Playing the 6 of Spade",
                23: "Playing the 6 of Diamond",
                24: "Playing the 7 of Heart",
                25: "Playing the 7 of Club",
                26: "Playing the 7 of Spade",
                27: "Playing the 7 of Diamond",
                28: "Playing the 8 of Heart",
                29: "Playing the 8 of Club",
                30: "Playing the 8 of Spade",
                31: "Playing the 8 of Diamond",
                32: "Playing the 9 of Heart",
                33: "Playing the 9 of Club",
                34: "Playing the 9 of Spade",
                35: "Playing the 9 of Diamond",
                36: "Playing the 10 of Heart",
                37: "Playing the 10 of Club",
                38: "Playing the 10 of Spade",
                39: "Playing the 10 of Diamond",
                40: "Playing the Payoo 1",
                41: "Playing the Payoo 2",
                42: "Playing the Payoo 3",
                43: "Playing the Payoo 4",
                44: "Playing the Payoo 5",
                45: "Playing the Payoo 6",
                46: "Playing the Payoo 7",
                47: "Playing the Payoo 8",
                48: "Playing the Payoo 9",
                49: "Playing the Payoo 10",
                50: "Playing the Payoo 11",
                51: "Playing the Payoo 12",
                52: "Playing the Payoo 13",
                53: "Playing the Payoo 14",
                54: "Playing the Payoo 15",
                55: "Playing the Payoo 16",
                56: "Playing the Payoo 17",
                57: "Playing the Payoo 18",
                58: "Playing the Payoo 19",
                59: "Playing the Payoo 20",
            }
            self.pos2act = {value: key for (key, value) in self.act2des.items()}


    def match(self,answer):
        pattern = r"Action: (.+)"
        match = re.search(pattern, answer)
        return match

    def extract_self_cards(self, info):
        card_num = 0
        cards = []
        for card in info:
            if card != -1:
                card_num += 1
                cards.append(card)
        return card_num, cards

    def extract_dun_cards(self, info):
        dun_cards = []

        for dun_card in info:
            if dun_card != -1:
                dun_cards.append(dun_card)

        return dun_cards

    def _get_board_status(self, player_id, observation):
        observation = observation['obs']
        self.cards_num, self.cards = self.extract_self_cards(observation[:15])
        self.score = observation[15]
        self.score_ops = observation[16:19]

        self.dun_cards = self.extract_dun_cards(observation[19:23])
        self.dun_color = observation[23]
        self.dun_num = observation[24]
        self.round_num = observation[25]
        self.papayoo = observation[26]

        self.player_id = player_id


    def _get_chat_action(
            self, action_mask, observation, player_id, info=None
    ):
        max_retries = 3
        current_retry = 0
        illegal_count = 0

        while current_retry < max_retries:
            try:
                
                self._get_board_status(player_id, observation=observation)
                
                message = f"Your observation now is :\n"

                if self.cards_num == 15:
                    message += "ATTENTION！A NEW ROUND HAS STARTED\n"

                message += f"You are Player [{self.player_id}] \n"

                message += f"You currently have {self.cards_num} cards, which are: \n"
                for idx, card in enumerate(self.cards):
                    message += f"({idx+1}). {card_id2des[card]} \n"
                message += f"your current score is {self.score} points \n"
                message += f"Player [{(self.player_id + 1) % 4}]'s current score is {self.score_ops[0]} points \n"
                message += f"Player [{(self.player_id + 2) % 4}]'s current score is {self.score_ops[1]} points \n"
                message += f"Player [{(self.player_id + 3) % 4}]'s current score is {self.score_ops[2]} points \n"

                if len(self.dun_cards) == 0:
                    message += "Currently, no cards have been played in this trick yet\n"
                else:
                    message += f"Currently, {len(self.dun_cards)} cards have been played in this trick. " \
                               f"The cards and the players who played them are:\n"
                    for idx in range(len(self.dun_cards)):
                        message += f"({idx+1}). Player [{(self.player_id - (len(self.dun_cards) - idx)) % 4}] " \
                                   f"played the {card_id2des[self.dun_cards[idx]]} in this trick\n"
                message += f"The current trick's suit is {color_id2des[self.dun_color]}\n"
                message += f"{self.dun_num} tricks have been completed in the current round\n"
                message += f"{self.round_num} rounds have been completed\n"
                message += f"The papayoo card of the current round is {card_id2des[self.papayoo]}. \n"

                message += "You should think step by step and output your action. For example: 'Playing the Ace of Heart'\n"
                message += "Your goal is to get as few cards with negative points as possible, such as Payoo and Papayoo.\n"
                message += f"Now you can choose one of the following actions:\n"
                for action in action_mask:
                    message += f"{self.act2des[action]} \n"
                message += f"You will respond with an action, formatted as:\n Action: <action>\n where you replace <action> with your actual action.\n"
                message += f"\nYou should explain why you choose the action\n"
                # print(message)
                answer = self._get_chat_answer(message=message)
                match = self.match(answer)
                if match:
                    pos = match.group(1)
                    pos = pos.rstrip('.')
                    # print(f"\n提取到的行动: {pos} \n")
                    action = self.pos2act.get(pos, -999)
                    # action == None代表匹配动作失败
                    if action == -999:
                        current_retry += 1
                        self.retry_counts += 1
                        self.match_two += 1
                        # print("匹配动作空间失败！")
                    # 表示匹配到了但是动作不合法
                    else:
                        if action in action_mask:
                            break
                        else: 
                            # print("It is an illegal action!")
                            current_retry += 1
                            self.retry_counts += 1
                            self.illegal_counts += 1
                else:
                    # print("匹配 Action:<action> 失败！")
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
            # print("random action:", action)

        return action
    
    def act(self, player_id, observation, info=None):
        
        # action_space = self.env.action_space
        action_space = list(range(59))
        legal_actions = observation['legal_actions']
        action_mask = list(legal_actions.keys())
        try:
            action = self._get_chat_action(action_mask, observation, player_id, info=None)
        except Exception as e:
            print(e)
            action = random.choice(action_mask)
            self.random_counts += 1
        return action  
