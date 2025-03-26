import numpy as np


class RandomAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = False

        self.retry_counts = 0
        self.match_one = 0
        self.match_two = 0
        self.illegal_counts = 0
        self.random_counts = 0


    def step(self, state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        self.random_counts += 1
        return np.random.choice(list(state['legal_actions'].keys()))
    
    def eval_step(self, observation):
        action = self.step(state=observation)
        return action, {}
    
    def set_env(self, env=None):
        if env is not None:
            self.env = env

    def _reset_count(self):
        self.retry_counts = 0
        self.match_one = 0
        self.match_two = 0
        self.illegal_counts = 0
        self.random_counts = 0
