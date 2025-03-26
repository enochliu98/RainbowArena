import re
import time
import random
from typing import Dict, Optional
from rainbowarena.agents.llm_agents.base_llm_agent import BaseLLMAgent

card_id2des = {
    0: "1",
    1: "2",
    2: "3",
    3: "4",
    4: "5",
    5: "6",
    6: "Lama",
}

class LamaAgent(BaseLLMAgent):

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
                0: "Playing 1",
                1: "Playing 2",
                2: "Playing 3",
                3: "Playing 4",
                4: "Playing 5",
                5: "Playing 6",
                6: "Playing Lama",
                7: "Drawing",
                8: "Passing",
            }
            self.pos2act = {value: key for (key, value) in self.act2des.items()}


    def match(self,answer):
        pattern = r"Action: (.+)"
        match = re.search(pattern, answer)
        return match

    def _get_board_status(self, observation):
        observation = observation['obs']
        self.cards_num = sum(observation[:7])
        self.cards = observation[:7]
        self.score = observation[7]
        self.quit = observation[8]
        self.score_op_1 = observation[9]
        self.quit_op_1 = observation[10]
        self.score_op_2 = observation[11]
        self.quit_op_2 = observation[12]
        self.deck_card = observation[13]

    def _get_chat_action(
            self, action_mask, observation, info=None
    ):
        max_retries = 3
        current_retry = 0

        while current_retry < max_retries:
            try:
                self._get_board_status(observation=observation)
                message = f"Your observation now is :\n"

                if self.cards_num == 13:
                    message += "ATTENTION！A NEW ROUND HAS STARTED\n"

                message += f"You currently have {self.cards_num} cards, as follows. For example, 'card 1 (amount: 3)' means there are 3 cards with the card 1 in your hand. \n"
                message += f"Card 1 (amount:{self.cards[0]}) \n" \
                           f"Card 2 (amount:{self.cards[1]}) \n" \
                           f"Card 3 (amount:{self.cards[2]}) \n" \
                           f"Card 4 (amount:{self.cards[3]}) \n" \
                           f"Card 5 (amount:{self.cards[4]}) \n" \
                           f"Card 6 (amount:{self.cards[5]}) \n" \
                           f"Card Lama (amount:{self.cards[6]}) \n"
                message += f"Your score is {self.score} \n"
                if self.quit == 1:
                    message += "You are out for the rest of the round \n"
                else:
                    message += "You are not out for the rest of the round \n"

                message += f"The next player's score is {self.score_op_1} \n"
                if self.quit_op_1 == 1:
                    message += "The next player is out for the rest of the round \n"
                else:
                    message += "The next player is not out for the rest of the round \n"

                message += f"The previous player's score is {self.score_op_2} \n"
                if self.quit_op_2 == 1:
                    message += "The previous player is out for the rest of the round \n"
                else:
                    message += "The previous player is not out for the rest of the round \n"

                message += f"The top card of the discard pile is {card_id2des[self.deck_card]} \n"

                message += "You should think step by step and output your action. \n"
                message += "For example, You can choose 'Playing 3' when the discard pile is '2' or '3' and can choose 'Drawing' when you have no card to play. Note! Choosing 'Passing' means quitting this round and waiting for other players to finish and then settle. \n"
                message += f"Now you can choose one of the following actions: \n"
                for index, action in enumerate(action_mask, start=1):
                    message += f"{index}. {self.act2des[action]}\n"
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
        action_space = list(range(8))
        legal_actions = observation['legal_actions']
        action_mask = list(legal_actions.keys())
        try:
            action = self._get_chat_action(action_mask, observation, info=None)
        except Exception as e:
            print(e)
            action = random.choice(action_mask)
            self.random_counts += 1
        return action
