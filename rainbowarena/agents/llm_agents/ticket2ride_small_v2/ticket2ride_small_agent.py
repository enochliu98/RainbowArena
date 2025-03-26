import openai 
import random
from typing import Dict, Optional
from pathlib import Path
import re
import time
from rainbowarena.agents.llm_agents.base_llm_agent import BaseLLMAgent


TICKET_CARDS = [
    [0, 0, 4, 4],
    [1, 0, 7, 5],
    [2, 1, 4, 3],
    [3, 1, 7, 5],
    [4, 1, 6, 4],
    [5, 2, 4, 3],
    [6, 3, 7, 4],
    [7, 5, 6, 2]
]

LINES = [
    [0, 3, 2, 0, 6, 0, 0],
    [1, 1, 4, 0, 2, 0, 0],
    [2, 2, 1, 0, 3, 0, 0],
    [3, 2, 3, 0, 1, 0, 0],
    [4, 1, 5, 1, 3, 0, 0],
    [5, 3, 4, 1, 5, 0, 0],
    [6, 1, 2, 2, 3, 0, 0],
    [7, 2, 1, 2, 6, 0, 0],
    [8, 3, 5, 3, 5, 0, 0],
    [9, 2, 4, 3, 4, 0, 0],
    [10, 3, 3, 3, 6, 0, 0],
    [11, 1, 3, 4, 5, 0, 0],
    [12, 1, 5, 4, 6, 0, 0],
    [13, 2, 1, 4, 7, 0, 0],
    [14, 2, 2, 5, 7, 0, 0],
    [15, 2, 5, 6, 7, 0, 0],
]

COLORS = {
    0: 'Rainbow',
    1: 'Yellow',
    2: 'Green',
    3: 'Blue',
    4: 'Red',
    5: 'No Restriction',
    -1: 'No Card'
}


class TicketAgent(BaseLLMAgent):
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
        self.match_one = 0
        self.match_two = 0
        self.retry_counts = 0
        self.random_counts = 0
        self.use_raw = False
        self.act2des = {
            0: "Take the Train Car Card from the face-up position (1) on the deck",
            1: "Take the Train Car Card from the face-up position (2) on the deck",
            2: "Take the Train Car Card from the face-up position (3) on the deck",
            3: "Take the Train Car Card from the face-up position (4) on the deck",
            4: "Take the Train Car Card from the face-up position (5) on the deck",
            5: "Draw one Train Car Card from the draw pile",
            6: "Claim route 0",
            7: "Claim route 1",
            8: "Claim route 2",
            9: "Claim route 3",
            10: "Claim route 4 using the Yellow Train Car Cards",
            11: "Claim route 4 using the Green Train Car Cards",
            12: "Claim route 4 using the Blue Train Car Cards",
            13: "Claim route 4 using the Red Train Car Cards",
            14: "Claim route 5",
            15: "Claim route 6",
            16: "Claim route 7",
            17: "Claim route 8 using the Yellow Train Car Cards",
            18: "Claim route 8 using the Green Train Car Cards",
            19: "Claim route 8 using the Blue Train Car Cards",
            20: "Claim route 8 using the Red Train Car Cards",
            21: "Claim route 9",
            22: "Claim route 10",
            23: "Claim route 11",
            24: "Claim route 12 using the Yellow Train Car Cards",
            25: "Claim route 12 using the Green Train Car Cards",
            26: "Claim route 12 using the Blue Train Car Cards",
            27: "Claim route 12 using the Red Train Car Cards",
            28: "Claim route 13",
            29: "Claim route 14",
            30: "Claim route 15 using the Yellow Train Car Cards",
            31: "Claim route 15 using the Green Train Car Cards",
            32: "Claim route 15 using the Blue Train Car Cards",
            33: "Claim route 15 using the Red Train Car Cards",
            34: "Declare the action to draw Destination Tickets",
            35: "Take the Destination Tickets from the face-up positions (1) on the deck",
            36: "Take the Destination Tickets from the face-up positions (2) on the deck",
            37: "Take the Destination Tickets from the face-up positions (3) on the deck",
            38: "Take the Destination Tickets from the face-up positions (1, 2) on the deck",
            39: "Take the Destination Tickets from the face-up positions (1, 3) on the deck",
            40: "Take the Destination Tickets from the face-up positions (2, 3) on the deck",
            41: "Take the Destination Tickets from the face-up positions (1, 2, 3) on the deck",
        }
        self.pos2act = {value:key for (key,value) in self.act2des.items()}
        
    def match(self,answer):
        pattern = r"Action: (.+)"
        match = re.search(pattern, answer)
        return match
   

    def extract_tickets(self, info):
        tickets = []
        for i in range(8):
            if info[i] == 1:
                tickets.append(TICKET_CARDS[i])
        return tickets

    def extract_routes(self, info):
        routes_self = []
        routes_opponent = []
        routes_free = []
        for i in range(16):
            if info[i] == 0:
                routes_free.append(LINES[i])
            elif info[i] == 1:
                routes_self.append(LINES[i])
            else:
                routes_opponent.append(LINES[i])

        return routes_self, routes_opponent, routes_free

    def extract_train_car_cards_deck(self, info):
        count = 0
        for train_car_card in info:
            if train_car_card == -1:
                count += 1
        return 5 - count, info

    def extract_tickets_deck(self, info):
        count = 0
        for ticket in info:
            if ticket == -1:
                count += 1
        return 3 - count, info

    def _get_board_status(self, observation):
        board_status = "\n"
        observation = observation['obs']
        self.train_pieces = observation[0]
        self.train_pieces_opponent = observation[1]
        self.train_car_cards = observation[2:7]
        self.tickets = self.extract_tickets(observation[7:15])
        self.routes_self, self.routes_opponent, self.routes_free = self.extract_routes(observation[15:31])
        self.score = observation[31]
        self.score_opponent = observation[32]
        self.train_car_cards_deck_count, self.train_car_cards_deck = self.extract_train_car_cards_deck(observation[33:38])
        self.tickets_deck_count, self.tickets_deck = self.extract_tickets_deck(observation[38:41])

    # Gather message, call the api and regularize the output
    def _get_chat_action(
            self, action_mask, observation, info=None
    ):
        max_retries = 3
        current_retry = 0
        illegal_count = 0

        while current_retry < max_retries:
            try:
                self._get_board_status(observation=observation)
                
                message = f"Your observation now is :\n"
                message += f"The number of your Train Pieces is : \n {self.train_pieces}\n"
                message += f"The number of your opponent's Train Pieces is : \n {self.train_pieces_opponent}\n"
                message += f"The number of each color of your Train Car Cards : \n" \
                           f"Rainbow ({self.train_car_cards[0]}), Yellow ({self.train_car_cards[1]}), " \
                           f"Green ({self.train_car_cards[2]}), Blue ({self.train_car_cards[3]}), " \
                           f"Red ({self.train_car_cards[4]}) \n"
                message += f"You currently have {len(self.tickets)} Destination Tickets, which are:\n"
                if len(self.tickets) == 0:
                    message += f"Empty \n"
                else:
                    for idx, ticket in enumerate(self.tickets):
                        message += f"[{idx}]. Start Station ({ticket[1]}), End Station ({ticket[2]}), Score ({ticket[3]}) \n"
                message += f"The current occupancy status of each Route on the Game Board is as follows: \n"
                message += f"You currently have occupied {len(self.routes_self)} routes, which are:\n"
                if len(self.routes_self) == 0:
                    message += f"Empty \n"
                else:
                    for idx, route in enumerate(self.routes_self):
                        message += f"[{idx}]. Route Number ({route[0]}), Length ({route[1]}), Color ({COLORS[route[2]]}), " \
                                   f"Start Station ({route[3]}), End Station ({route[4]}) \n"
                message += f"Your opponent currently have occupied {len(self.routes_opponent)} routes, which are:\n"
                if len(self.routes_opponent) == 0:
                    message += f"Empty \n"
                else:
                    for idx, route in enumerate(self.routes_opponent):
                        message += f"[{idx}]. Route Number ({route[0]}), Length ({route[1]}), Color ({COLORS[route[2]]}), " \
                                   f"Start Station ({route[3]}), End Station ({route[4]}) \n"
                message += f"There are currently {len(self.routes_free)} routes that have not been occupied yet, which are:\n"
                if len(self.routes_free) == 0:
                    message += f"Empty \n"
                else:
                    for idx, route in enumerate(self.routes_free):
                        message += f"[{idx}]. Route Number ({route[0]}), Length ({route[1]}), Color ({COLORS[route[2]]}), " \
                                   f"Start Station ({route[3]}), End Station ({route[4]}) \n"
                message += f"Your score is {self.score}\n"
                message += f"Your opponent's score is {self.score_opponent}\n"
                message += f"There are totally {self.train_car_cards_deck_count} Train Car Cards on the deck, which are:\n"
                for idx, train_car_card in enumerate(self.train_car_cards_deck):
                    message += f"Position {idx+1}. {COLORS[self.train_car_cards_deck[idx]]}"
                message += f"There are totally {self.tickets_deck_count} visible Destination Tickets on the deck, which are: \n"
                for idx, ticket in enumerate(self.tickets_deck):
                    if ticket != -1:
                        ticket = TICKET_CARDS[ticket]
                        message += f"Position {idx+1}. Start Station ({ticket[1]}), End Station ({ticket[2]}), Score ({ticket[3]}) \n"
                    else:
                        message += f"Position {idx+1}. the ticket is invisible or there is no ticket in this position\n"

                message += "You should think step by step and output your action. For example: 'Take the Train Car Card from the face-up position (1) on the deck '\n"
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
        action_space = list(range(41))
        action_mask = observation['raw_legal_actions']
        try:
            action = self._get_chat_action(action_mask, observation, info=None)
        except Exception as e:
            print(e)
            action = random.choice(action_space)
            self.random_counts += 1
        return action  