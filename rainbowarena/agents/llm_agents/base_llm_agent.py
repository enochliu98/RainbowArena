import openai 
from openai import OpenAI
import random
from typing import Dict, Optional
from pathlib import Path
import re
from abc import ABC, abstractmethod
class BaseLLMAgent(ABC):
    def __init__(
        self,
        llm_info: Optional[Dict[str, str]] = None,
    ):
        self.llm_info = llm_info
        self.stop_token = None
        self.need_history = False
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

    @abstractmethod  
    def match(self,answer):
        pattern = r"Action: (.+)"
        match = re.search(pattern, answer)
        return match
   
    @abstractmethod
    # Gather message, call the api and regularize the output
    def _get_chat_action(
            self,action_mask, observation, info = None
    ):
        indices_of_ones = [i for i, value in enumerate(action_mask) if value == 1]

        if not indices_of_ones:
            return None
        return random.choice(indices_of_ones)
    
    # 
    def act(self, player_id, observation, reward, termination, info):
        action_mask = observation.get("action_mask", None)
        try:
            action = self._get_chat_action(action_mask, observation, info)
        except Exception as e:
            print(e)
        return action
    
    def eval_step(self, observation):
        action = self.act(player_id = self.player_id, observation=observation)
        return action, {}
    
    # call api
    def _get_chat_answer(self,message):
        self.messages.append({"role":"user","content":message})
        api_kwargs = dict(api_key=self.api_key, base_url = self.api_base)
        client = OpenAI(**api_kwargs)
        if self.stop_token is not None:
            answer = client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature = self.temperature,
                stop = [self.stop_token]
            )
        else:
            answer = client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature = self.temperature,
            )
        answer = answer.choices[0].message.content
        # print(answer)
        if self.need_history:
            self.messages.append({"role":"assistant","content":answer})
        else:
            self._reset_message()
        return answer
    
    def _reset_message(self):
        self.messages =  [{"role": "system", "content": self.prompt_rule}]

    def set_env(self, env=None):
        if env is not None:
            self.env = env

    def set_id(self, player_id):
        if player_id is not None:
            self.player_id = player_id

    def _reset_count(self):
        self.retry_counts = 0
        self.match_one = 0
        self.match_two = 0
        self.illegal_counts = 0
        self.random_counts = 0