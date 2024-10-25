import argparse, os
import random
import numpy as np
import h5py
import torch
import d4rl, gym
import yaml
import re

import rlkit.torch.pytorch_util as ptu
# from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.torch.sac.policies import VAEPolicy, MakeDeterministic
from rlkit.torch.data_management.replay_buffer import BatchReplayBuffer
from rlkit.torch.sac.bc_vae import BCTrainer
from rlkit.torch.data_management.dataset import sarsa_dataset
# import rlkit.envs.mujoco_env_new

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_random_seed(seed):
    print(f'Set random seed {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_hdf5(dataset, replay_buffer):
    replay_buffer._obs = dataset['observations']
    replay_buffer._next_obs = dataset['next_observations']
    replay_buffer._actions = dataset['actions']
    replay_buffer._next_actions = dataset['next_actions']
    replay_buffer._rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer._terminals = np.expand_dims(np.squeeze(dataset['terminals']), 1)
    replay_buffer._size = dataset['terminals'].shape[0]
    print('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size

def load_state_dict_data(base_dir, algo_name, env_name, seed, itr=None):
    p = re.compile(f's{seed}' + '_\d{6}_\d{6}')

    filename = f'itr_{itr}.pkl' if itr else 'params.pkl'

    file = None
    run_dir = os.path.join(base_dir, algo_name, env_name)

    for dir in os.listdir(run_dir):
        m = p.match(dir)
        if m:
            file_t = os.path.join(run_dir, dir, filename)
            if os.path.isfile(file_t):
                file = file_t
                break

    if file is None:
        print(f'There is no such file {filename} in the directory {run_dir}')
        print(f'Seed : {seed}, Env : {env_name}')
        return None
        # raise FileNotFoundError

    snapshot = torch.load(file)
    print(f'{file} is loaded successfully')

    trainer_snapshot = {}
    for k, v in snapshot.items():
        if k.startswith('trainer/'):
            trainer_snapshot[k[8:]] = v

    return trainer_snapshot


def experiment(variant):
    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )

    policy = VAEPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M],
        latent_dim=obs_dim * 2,
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector(
        eval_env,
    )

    replay_buffer = BatchReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    if variant['load_buffer']:
        load_hdf5(sarsa_dataset(eval_env), replay_buffer)

    replay_buffer.calculate_gamma_return(gamma=variant['trainer_kwargs']['discount'])

    trainer = BCTrainer(
        env=eval_env,
        policy=policy,
        qf=qf,
        target_qf=target_qf,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_both=False,
        batch_rl=variant['load_buffer'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)

    algorithm.train()


def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='halfcheetah-medium-v2')
    parser.add_argument("--exp_name", type=str, default='VAE_bc')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--policy", type=str, default='tanh')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    enable_gpus(args.gpu)

    with open(f'./run_examples/VAE_bc/{args.env}.yaml') as f:
        variant = yaml.safe_load(f)

    variant['buffer_filename'] = None
    variant['load_buffer'] = True

    variant['policy'] = args.policy
    variant['env_name'] = args.env
    variant['seed'] = args.seed

    # Set seed and logger
    set_random_seed(variant['seed'])

    print('Base_Log_Dir: ', BASE_DIR)
    variant['model_dir'] = BASE_DIR

    print(variant['logger_kwargs'])
    variant['exp_name'] = exp_name = args.exp_name
    setup_logger(exp_name, env_name=variant['env_name'], seed=variant['seed'],
                 variant=variant, base_log_dir=BASE_DIR,
                 include_exp_prefix_sub_dir=False, **variant['logger_kwargs'])

    ptu.set_gpu_mode(True)
    experiment(variant)
