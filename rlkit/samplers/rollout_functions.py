import numpy as np
import rlkit.torch.pytorch_util as ptu
from tqdm import tqdm
import torch

def multitask_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
    goal = o[desired_goal_key]
    while path_length < max_path_length:
        dict_obs.append(o)
        if observation_key:
            o = o[observation_key]
        new_obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(goal[None], path_length, 0),
        full_observations=dict_obs,
    )


def rollout(
        env,
        agent,
        mean=0.0,
        std=1.0,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        if isinstance(o, tuple):
            o = o[0]
        a, agent_info = agent.get_action(o)

        next_o, r, d, env_info = env.step(a)
        # next_o, r, d, env_info, *_ = env.step(a)

        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])

    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )

    # observations = (observations - mean) / std
    # next_observations = (next_observations - mean) / std

    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def batcheval_rollout(
        env,
        agent,
        qf1,
        qf2,
        init_obs,
        init_actions,
        gamma=0.99,
        N_episodes=10,
        max_path_length=1000,
):

    for j in range(len(init_actions)):

        long_bat_returns = []
        bat_returns = []
        q_vals = []
        long_gam_return = 0

        for n in range(N_episodes):

            rewards = []
            obs = []
            actions = []

            o = env.reset()
            agent.reset()
            d = False
            next_o = None

            init_s = True
            t = 0
            while t < max_path_length or d != True:

                if init_s:
                    a = init_actions[j]
                    init_s = False
                else:
                    a, agent_info = agent.get_action(o)

                # next_o, r, d, env_info, *_ = env.step(a)
                next_o, r, d, env_info = env.step(a)
                obs.append(o)
                actions.append(a)
                rewards.append(r)

                if d:
                    next_o = env.reset()

                t += 1
                o = next_o

            gam_return = 0
            for i in reversed(range(len(rewards))):
                gam_return = rewards[i] + gamma * gam_return  # * (1 - terminals[i])
                long_gam_return = rewards[i] + gamma * long_gam_return  # * (1 - terminals[i])

                if n == 1:
                    long_bat_returns.append(long_gam_return)

                    with torch.no_grad():
                        q_val = torch.stack([qf1(ptu.from_numpy(obs[i]).unsqueeze(0),
                                                      ptu.from_numpy(actions[i]).unsqueeze(0)),
                                             qf2(ptu.from_numpy(obs[i]).unsqueeze(0),
                                                      ptu.from_numpy(actions[i]).unsqueeze(0))], 0)
                        q_val = torch.min(q_val, dim=0)[0].squeeze().cpu().numpy()

                    q_vals.append(q_val)
            bat_returns.append(gam_return)


    return np.array(bat_returns).squeeze(), np.array(q_vals), np.array(long_bat_returns)

def function_rollout(
        env,
        agent_fn,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        env_mean=0.0,
        env_std=1.0,
):

    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    env_infos = []
    o = env.reset()
    o = (o - env_mean) / env_std
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a = agent_fn(o)
        next_o, r, d, env_info = env.step(a)
        next_o = (next_o - env_mean) / env_std
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        env_infos=env_infos,
    )

