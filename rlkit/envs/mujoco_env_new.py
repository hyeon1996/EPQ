from gym.envs.registration import register
import d4rl
from d4rl.gym_mujoco import gym_envs
from d4rl import infos


NEW_DATASET_URLS = {
    'halfcheetah-expert-medium-random-v2': '~/.d4rl/datasets/halfcheetah_expert_medium_random-v2.hdf5',
    'walker2d-expert-medium-random-v2': '~/.d4rl/datasets/walker2d_expert_medium_random-v2.hdf5',
}

NEW_REF_MIN_SCORE = {
    'halfcheetah-expert-medium-random-v2': -280.178953,
    'walker2d-expert-medium-random-v2': 1.629008,
}

NEW_REF_MAX_SCORE = {
    'halfcheetah-expert-medium-random-v2': 12135.0,
    'walker2d-expert-medium-random-v2': 4592.3,
}


register(
    id='halfcheetah-expert-medium-random-v2',
    entry_point='d4rl.gym_mujoco.gym_envs:get_cheetah_env',
    max_episode_steps=1000,
    kwargs={
        'deprecated': False,
        'ref_min_score': NEW_REF_MIN_SCORE['halfcheetah-expert-medium-random-v2'],
        'ref_max_score': NEW_REF_MAX_SCORE['halfcheetah-expert-medium-random-v2'],
        'dataset_url': NEW_DATASET_URLS['halfcheetah-expert-medium-random-v2'],
        'h5path': None,
    }
)

register(
    id='walker2d-expert-medium-random-v2',
    entry_point='d4rl.gym_mujoco.gym_envs:get_walker_env',
    max_episode_steps=1000,
    kwargs={
        'deprecated': False,
        'ref_min_score': NEW_REF_MIN_SCORE['walker2d-expert-medium-random-v2'],
        'ref_max_score': NEW_REF_MAX_SCORE['walker2d-expert-medium-random-v2'],
        'dataset_url': NEW_DATASET_URLS['walker2d-expert-medium-random-v2'],
        'h5path': None,
    }
)
