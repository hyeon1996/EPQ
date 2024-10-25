import argparse, os
import random
import numpy as np
import torch
import d4rl, gym
import re
import pickle
import yaml

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger

from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import GaussianPolicy, TanhGaussianPolicy, VAEPolicy, MakeDeterministic
from rlkit.torch.sac.epq_trainer import CQLTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from clustering import nearest_neighborhood_cluster

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
        print(f"Is GPU available : {torch.cuda.is_available()}")
        torch.cuda.manual_seed(seed)

def load_state_dict_data(base_dir, algo_name, env_name, seed, itr=None):
    p = re.compile(f's{seed}' + '_\d{6}_\d{6}')

    filename = f'itr_{itr}.pkl' if itr else 'params.pkl'

    file = None
    run_dir = os.path.join(base_dir, algo_name, env_name)

    try:
        for dir in os.listdir(run_dir):
            m = p.match(dir)
            if m:
                file_t = os.path.join(run_dir, dir, filename)
                if os.path.isfile(file_t):
                    file = file_t
                    break
    except:
        return None

    if file is None:
        print(f'There is no such file {filename} in the directory {run_dir}')
        print(f'Seed : {seed}, Env : {env_name}')
        return None

    snapshot = torch.load(file)
    print(f'{file} is loaded successfully')

    trainer_snapshot = {}
    for k, v in snapshot.items():
        if k.startswith('trainer/'):
            trainer_snapshot[k[8:]] = v

    return trainer_snapshot


def load_hdf5(dataset, replay_buffer):
    replay_buffer._observations = dataset['observations']
    replay_buffer._next_obs = dataset['next_observations']
    replay_buffer._actions = dataset['actions']
    replay_buffer._rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer._terminals = np.expand_dims(np.squeeze(dataset['terminals']), 1)
    replay_buffer._size = dataset['terminals'].shape[0]
    print('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size


def experiment(variant, snapshot_data):
    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env

    if snapshot_data is not None:
        print("snapshot_data is not None")
        if snapshot_data['vae'] is not None:
            vae_data = snapshot_data['vae']
        else:
            vae_data = None

        if snapshot_data['qf'] is not None and variant['use_q_beta']:
            qf_data = snapshot_data['qf']
        else:
            qf_data = None

    else:
        print("snapshot_data is None")
        vae_data = None
        qf_data = None

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )

    if variant['policy'] == 'tanh':
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M, M],
        )
    elif variant['policy'] == 'normal':
        policy = GaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M, M],
        )

    vae = VAEPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M],
        latent_dim=obs_dim * 2,
    )
    if vae_data is not None:
        vae.load_state_dict(vae_data.state_dict())

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector(
        eval_env,
    )

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    load_hdf5(d4rl.qlearning_dataset(eval_env), replay_buffer)
    replay_buffer.init_qf(qf_data= qf_data, gamma=variant['trainer_kwargs']['discount'])

    file_name = f'./NN_data/nn_data_info' + '_%s' % str(variant['env_name']) + '.pickle'

    if os.path.exists(file_name):
        with open(file_name, 'rb') as fr:
            clustering_info_loaded = pickle.load(fr)
        print(file_name)
        print("Cluster info loaded !")
    else :
        print("Do Clustering")
        clustering_info_loaded = nearest_neighborhood_cluster(variant['env_name'], replay_buffer, n_neighbors=variant['nn'])

    dist = clustering_info_loaded['distances']
    idxs = clustering_info_loaded['idxs']

    adj_dist = np.mean([np.min(dist[i][dist[i]>1e-5]) for i in range(len(dist))])

    d=variant['epsilon']
    cluster_idx_list = [idxs[i][dist[i] <= d * adj_dist] for i in range(len(dist))]
    print("Cluster distance : %f"%(d * adj_dist))

    trainer = CQLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vae=vae,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs'],
    )

    replay_buffer.calculate_is_weight(cluster_idx_list, temp=variant['zeta'])

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
    parser.add_argument("--exp_name", type=str, default='EPQ')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--policy", type=str, default='tanh')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    enable_gpus(args.gpu)

    with open(f'./run_examples/EPQ/{args.env}.yaml') as f:
        variant = yaml.safe_load(f)

    variant['buffer_filename'] = None

    variant['layer_size'] = 256 if 'v2' in args.env else 1024
    variant['load_buffer'] = True
    variant['policy'] = args.policy
    variant['env_name'] = args.env
    variant['seed'] = args.seed if args.seed >= 0 else random.randint(0, 10000000)

    variant['trainer_kwargs']['raw_action'] = True if variant['policy'] == 'tanh' else False
    print("="*50)
    print("Seed %d"%variant['seed'])
    print("="*50)

    print('Base_Log_Dir: ', BASE_DIR)

    snapshot_data = load_state_dict_data(
        base_dir=BASE_DIR, env_name=variant['env_name'], algo_name="VAE_bc",
        seed=0, itr=2000,
    )

    exp_name = args.exp_name

    set_random_seed(variant['seed'])

    setup_logger(exp_name, env_name=variant['env_name'], seed=variant['seed'], variant=variant, base_log_dir=BASE_DIR, include_exp_prefix_sub_dir=False, **variant['logger_kwargs'])
    ptu.set_gpu_mode(True)

    experiment(variant, snapshot_data)
