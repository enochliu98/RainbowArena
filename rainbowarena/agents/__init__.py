import subprocess
import sys
from distutils.version import LooseVersion

reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

if 'torch' in installed_packages:
    from rainbowarena.agents.doubledqn_agent import DQNAgent as DoubleDQNAgent
    from rainbowarena.agents.nfsp_agent import NFSPAgent as NFSPAgent
    from rainbowarena.agents.dqnv1_agent import DQNV1Agent
    from rainbowarena.agents.dqnv2_agent import DQNV2Agent
    from rainbowarena.agents.dqnv3_agent import DQNV3Agent
    from rainbowarena.agents.dqnv4_agent import DQNV4Agent
    from rainbowarena.agents.ppo_agent import PPOAgent as PPOAgent
    from rainbowarena.agents.dqn_agent import DQNAgent as DQNAgent
    from rainbowarena.agents.duelingdqn_agent import DQNAgent as DuelingDQNAgent
    from rainbowarena.agents.noisydqn_agent import DQNAgent as NoisyDQNAgent
    from rainbowarena.agents.categoricaldqn_agent import DQNAgent as CategoricalDQNAgent
    from rainbowarena.agents.om_agent import DQNAgent as OMAgent

from rainbowarena.agents.cfr_agent import CFRAgent
from rainbowarena.agents.human_agents.limit_holdem_human_agent import HumanAgent as LimitholdemHumanAgent
from rainbowarena.agents.human_agents.nolimit_holdem_human_agent import HumanAgent as NolimitholdemHumanAgent
from rainbowarena.agents.human_agents.leduc_holdem_human_agent import HumanAgent as LeducholdemHumanAgent
from rainbowarena.agents.human_agents.blackjack_human_agent import HumanAgent as BlackjackHumanAgent
from rainbowarena.agents.human_agents.uno_human_agent import HumanAgent as UnoHumanAgent
from rainbowarena.agents.random_agent import RandomAgent

