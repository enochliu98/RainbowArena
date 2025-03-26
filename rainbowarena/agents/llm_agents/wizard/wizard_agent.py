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
    40: "11 of Heart",
    41: "11 of Club",
    42: "11 of Spade",
    43: "11 of Diamond",
    44: "12 of Heart",
    45: "12 of Club",
    46: "12 of Spade",
    47: "12 of Diamond",
    48: "13 of Heart",
    49: "13 of Club",
    50: "13 of Spade",
    51: "13 of Diamond",
    52: "Wizard 1",
    53: "Wizard 2",
    54: "Wizard 3",
    55: "Wizard 4",
    56: "Jester 1",
    57: "Jester 2",
    58: "Jester 3",
    59: "Jester 4",
}

color_id2des1 = {
    -1: '[The trump suit has not been determined yet]',
    0: 'Heart',
    1: 'Club',
    2: 'Spade',
    3: 'Diamond',
    4: '[No trump suit]'
}

color_id2des2 = {
    -1: '[The suit has not been determined yet]',
    0: 'Heart',
    1: 'Club',
    2: 'Spade',
    3: 'Diamond',
    4: 'Wizard'
}

class WizardAgent(BaseLLMAgent):

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
                0: "Selecting [Heart] as the trump suit",
                1: "Selecting [Club] as the trump suit",
                2: "Selecting [Spade] as the trump suit",
                3: "Selecting [Diamond] as the trump suit",
                4: "Predicting: winning [0] tricks in this round",
                5: "Predicting: winning [1] trick in this round",
                6: "Predicting: winning [2] tricks in this round",
                7: "Predicting: winning [3] tricks in this round",
                8: "Predicting: winning [4] tricks in this round",
                9: "Predicting: winning [5] tricks in this round",
                10: "Predicting: winning [6] tricks in this round",
                11: "Predicting: winning [7] tricks in this round",
                12: "Predicting: winning [8] tricks in this round",
                13: "Predicting: winning [9] tricks in this round",
                14: "Predicting: winning [10] tricks in this round",
                15: "Predicting: winning [11] tricks in this round",
                16: "Predicting: winning [12] tricks in this round",
                17: "Predicting: winning [13] tricks in this round",
                18: "Predicting: winning [14] tricks in this round",
                19: "Predicting: winning [15] tricks in this round",
                20: "Playing the 1 of Heart",
                21: "Playing the 1 of Club",
                22: "Playing the 1 of Spade",
                23: "Playing the 1 of Diamond",
                24: "Playing the 2 of Heart",
                25: "Playing the 2 of Club",
                26: "Playing the 2 of Spade",
                27: "Playing the 2 of Diamond",
                28: "Playing the 3 of Heart",
                29: "Playing the 3 of Club",
                30: "Playing the 3 of Spade",
                31: "Playing the 3 of Diamond",
                32: "Playing the 4 of Heart",
                33: "Playing the 4 of Club",
                34: "Playing the 4 of Spade",
                35: "Playing the 4 of Diamond",
                36: "Playing the 5 of Heart",
                37: "Playing the 5 of Club",
                38: "Playing the 5 of Spade",
                39: "Playing the 5 of Diamond",
                40: "Playing the 6 of Heart",
                41: "Playing the 6 of Club",
                42: "Playing the 6 of Spade",
                43: "Playing the 6 of Diamond",
                44: "Playing the 7 of Heart",
                45: "Playing the 7 of Club",
                46: "Playing the 7 of Spade",
                47: "Playing the 7 of Diamond",
                48: "Playing the 8 of Heart",
                49: "Playing the 8 of Club",
                50: "Playing the 8 of Spade",
                51: "Playing the 8 of Diamond",
                52: "Playing the 9 of Heart",
                53: "Playing the 9 of Club",
                54: "Playing the 9 of Spade",
                55: "Playing the 9 of Diamond",
                56: "Playing the 10 of Heart",
                57: "Playing the 10 of Club",
                58: "Playing the 10 of Spade",
                59: "Playing the 10 of Diamond",
                60: "Playing the 11 of Heart",
                61: "Playing the 11 of Club",
                62: "Playing the 11 of Spade",
                63: "Playing the 11 of Diamond",
                64: "Playing the 12 of Heart",
                65: "Playing the 12 of Club",
                66: "Playing the 12 of Spade",
                67: "Playing the 12 of Diamond",
                68: "Playing the 13 of Heart",
                69: "Playing the 13 of Club",
                70: "Playing the 13 of Spade",
                71: "Playing the 13 of Diamond",
                72: "Playing the Wizard 1",
                73: "Playing the Wizard 2",
                74: "Playing the Wizard 3",
                75: "Playing the Wizard 4",
                76: "Playing the Jester 1",
                77: "Playing the Jester 2",
                78: "Playing the Jester 3",
                79: "Playing the Jester 4",

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
        self.duns = observation[16]
        self.duns_prd = observation[17]
        self.score_ops = observation[18:21]
        self.duns_ops = observation[21:24]
        self.duns_prd_ops = observation[24:27]

        self.deck_ace_color = observation[27]
        self.deck_dun_color = observation[28]
        self.dun_cards = self.extract_dun_cards(observation[29:33])
        self.dun_num = observation[33]
        self.round_num = observation[34]

        self.player_id = player_id


    def _get_chat_action(
            self, action_mask, observation, player_id, info=None
    ):
        max_retries = 3
        current_retry = 0

        while current_retry < max_retries:
            try:
                self._get_board_status(player_id, observation=observation)

                message = f"Your observation now is :\n"

                message += f"Attention: It is the {self.dun_num + 1}th trick (Totally {self.round_num + 1} tricks in this round) " \
                           f"of the {self.round_num + 1}th round (Totally 15 rounds in this game)\n"

                message += f"You are Player [{self.player_id}] \n"

                message += f"You currently have {self.cards_num} cards, which are: \n"
                for idx, card in enumerate(self.cards):
                    message += f"({idx+1}). {card_id2des[card]} \n"
                message += f"your current score is {self.score} points \n"
                message += f"Player [{(self.player_id + 1) % 4}]'s current score is {self.score_ops[0]} points \n"
                message += f"Player [{(self.player_id + 2) % 4}]'s current score is {self.score_ops[1]} points \n"
                message += f"Player [{(self.player_id + 3) % 4}]'s current score is {self.score_ops[2]} points \n"
                
                message += f"In the current round, you predicted that you would win {self.duns} tricks. " \
                           f"So far, you have won {self.duns_prd} tricks\n"
                message += f"In the current round, Player [{(self.player_id + 1) % 4}] " \
                           f"predicted that he would win {self.duns_prd_ops[0]} tricks. " \
                           f"So far, he has won {self.duns_ops[0]} tricks\n"
                message += f"In the current round, Player [{(self.player_id + 2) % 4}] " \
                           f"predicted that he would win {self.duns_prd_ops[1]} tricks. " \
                           f"So far, he has won {self.duns_ops[1]} tricks\n"
                message += f"In the current round, Player [{(self.player_id + 3) % 4}] " \
                           f"predicted that he would win {self.duns_prd_ops[2]} tricks. " \
                           f"So far, he has won {self.duns_ops[2]} tricks\n"
                
                message += f"The trump suit of the current round is {color_id2des1[self.deck_ace_color]}\n"
                message += f"The suit of the current round is {color_id2des2[self.deck_dun_color]}\n"

                if len(self.dun_cards) == 0:
                    message += "Currently, no cards have been played in this trick yet\n"
                else:
                    message += f"Currently, {len(self.dun_cards)} cards have been played in this trick. " \
                               f"The cards and the players who played them are:\n"
                    for idx in range(len(self.dun_cards)):
                        message += f"({idx+1}). Player [{(self.player_id - (len(self.dun_cards) - idx)) % 4}] " \
                                   f"played the {card_id2des[self.dun_cards[idx]]} in this trick\n"
                
                message += "You should think step by step and output your action. For example: 'Playing the Ace of Heart'\n"
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
        action_space = list(range(79))
        legal_actions = observation['legal_actions']
        action_mask = list(legal_actions.keys())
        try:
            action = self._get_chat_action(action_mask, observation, player_id, info=None)
        except Exception as e:
            print(e)
            action = random.choice(action_mask)
            self.random_counts += 1
        return action  
