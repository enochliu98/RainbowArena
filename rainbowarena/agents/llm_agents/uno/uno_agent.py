import openai 
import random
from typing import Dict, Optional
from pathlib import Path
import re
import time
# from base_llm_agent import BaseLLMAgent
from rainbowarena.agents.llm_agents.base_llm_agent import BaseLLMAgent


class UnoAgent(BaseLLMAgent):
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
            self.match_one = 0
            self.match_two = 0
            self.retry_counts = 0
            self.random_counts = 0
            self.use_raw = False

            # Define colors and their corresponding offsets
            colors = {
                "r": 0,
                "g": 15,
                "b": 30,
                "y": 45,
            }
            # Generate number cards and action cards for each color
            for color, offset in colors.items():
                self.act2des.update({
                    **{i + offset: f"{color}-{i}" for i in range(10)},
                    10 + offset: f"{color}-skip",
                    11 + offset: f"{color}-reverse",
                    12 + offset: f"{color}-draw_2",
                    13 + offset: f"{color}-wild",
                    14 + offset: f"{color}-wild_draw_4",
                })
            self.act2des.update({60: "draw",})
            self.pos2act = {value:key for (key,value) in self.act2des.items()}

    def match(self,answer):
        pattern = r"Action: (.+)"
        match = re.search(pattern, answer)
        return match

    def _get_board_status(self, observation):
        
        self.hand = observation['raw_obs']['hand']
        self.target = observation['raw_obs']['played_cards'][-1]
        self.played_cards = observation['raw_obs']['played_cards']
        self.legal_actions = observation['raw_obs']['legal_actions']

    def _get_chat_action(self, action_mask, observation, info= None):
        max_retries = 3
        current_retry = 0
        
        while current_retry < max_retries:
            try:
                self._get_board_status(observation=observation)
                message = "Each card is represented as a string of color and trait(number, symbol/action). ‘r’, ‘b’, ‘y’, ‘g’ represent red, blue, yellow and green respectively. \n"
                message += f"Your observation now is :\n"
                message += f"Your hand now is {self.hand}.\n"
                message += f"Your target now is {self.target}.\n"
                message += f"The played cards are {self.played_cards}.\n"

                message += "You should think step by step and output your action. For example: 'r-6'.\n"
                message += f"Now you can and only choose one of the following legal actions:\n"
                message += f"{observation['raw_legal_actions']} \n"
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
        action_space = list(range(60))
        legal_actions = observation['legal_actions']
        action_mask = list(legal_actions.keys())
        try:
            action = self._get_chat_action(action_mask, observation, info=None)
        except Exception as e:
            print(e)
            action = random.choice(action_mask)
            self.random_counts += 1
        return action  