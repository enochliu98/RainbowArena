''' Wrrapers of pretrained models.
'''
import os

import rainbowarena
from rainbowarena.agents import CFRAgent
from rainbowarena.models.model import Model

# Root path of pretrianed models
ROOT_PATH = os.path.join(rainbowarena.__path__[0], 'models/pretrained')

class LeducHoldemCFRModel(Model):
    ''' A pretrained model on Leduc Holdem with CFR (chance sampling)
    '''
    def __init__(self):
        ''' Load pretrained model
        '''
        env = rainbowarena.make('leduc-holdem')
        self.agent = CFRAgent(env, model_path=os.path.join(ROOT_PATH, 'leduc_holdem_cfr'))
        self.agent.load()
    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return [self.agent, self.agent]

