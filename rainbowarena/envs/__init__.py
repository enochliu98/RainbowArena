''' Register new environments
'''
from rainbowarena.envs.env import Env
from rainbowarena.envs.registration import register, make


print("test")

register(
    env_id='blackjack',
    entry_point='rlcard.envs.blackjack:BlackjackEnv',
)

register(
    env_id='doudizhu',
    entry_point='rlcard.envs.doudizhu:DoudizhuEnv',
)

register(
    env_id='limit-holdem',
    entry_point='rlcard.envs.limitholdem:LimitholdemEnv',
)

register(
    env_id='no-limit-holdem',
    entry_point='rlcard.envs.nolimitholdem:NolimitholdemEnv',
)

register(
    env_id='leduc-holdem',
    entry_point='rlcard.envs.leducholdem:LeducholdemEnv'
)

register(
    env_id='uno',
    entry_point='rlcard.envs.uno:UnoEnv',
)

register(
    env_id='mahjong',
    entry_point='rlcard.envs.mahjong:MahjongEnv',
)

register(
    env_id='gin-rummy',
    entry_point='rlcard.envs.gin_rummy:GinRummyEnv',
)

register(
    env_id='bridge',
    entry_point='rlcard.envs.bridge:BridgeEnv',
)

register(
    env_id='splendor',
    entry_point='rlcard.envs.splendor:SplendorEnv'
)

register(
    env_id='splendor_v2',
    entry_point='rlcard.envs.splendor_v2:SplendorEnv'
)

register(
    env_id='ticket2ride_small_v2',
    entry_point='rlcard.envs.ticket2ride_small_v2:Ticket2RideEnv'
)

register(
    env_id='century',
    entry_point='rlcard.envs.century:CenturyEnv'
)

register(
    env_id='lama',
    entry_point='rlcard.envs.lama:LamaEnv'
)

register(
    env_id='wizard',
    entry_point='rlcard.envs.wizard:WizardEnv'
)

register(
    env_id='gongzhu',
    entry_point='rlcard.envs.gongzhu:GongzhuEnv'
)

register(
    env_id='gongzhu_v2',
    entry_point='rlcard.envs.gongzhu_v2:GongzhuEnv'
)

register(
    env_id='papayoo',
    entry_point='rlcard.envs.papayoo:PapayooEnv'
)