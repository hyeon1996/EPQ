import argparse, os
import numpy as np
import h5py
from tqdm import tqdm
import d4rl, gym
import d4rl.offline_env

from d4rl.utils.dataset_utils import DatasetWriter
from d4rl.offline_env import download_dataset_from_url, get_keys, DATASET_PATH
from run_examples.cql_mujoco_new import load_hdf5


def check_last_is_done(data_dict):
    print('last el in data_dict[terminals]: ', data_dict['terminals'][-1])
    return data_dict['terminals'][-1]


def check_timeout(data_dict):
    if 'timeouts' in data_dict:
        print('data_dict[timeouts]: ', data_dict['timeouts'][:20])
        return data_dict['timeouts'][:20]
    else:
        return None


def merge_dataset(dataset_list, dataset_dir='~/.d4rl/datasets'):
    if not isinstance(dataset_list, list):
        if isinstance(dataset_list, str):
            dataset_list = [dataset_list]
        else:
            print('Please put str in dataset_list')
            raise NotImplementedError

    gym_env_names = np.unique([env_name.split('-')[0] for env_name in dataset_list])
    gym_env_versions = np.unique([env_name.split('-')[-1] for env_name in dataset_list])
    gym_dataset_name = '_'.join([''.join(env_name.split('-')[1:-1]) for env_name in dataset_list])
    assert gym_env_names.shape[0] == 1, print('environment should be same')
    assert gym_env_versions.shape[0] == 1, print('environment version should be same')
    gym_env_name = gym_env_names[0]
    gym_env_version = gym_env_versions[0]

    merged_data = {}
    for i, env_name in enumerate(dataset_list):
        env = gym.make(env_name)
        data_dict = env.get_dataset()
        if 'timeouts' in data_dict:
            data_dict['timeouts'][-1] = True

        for k, v in data_dict.items():
            if not k.startswith('metadata') and not k.startswith('infos/action_log_probs'):
                if k not in merged_data.keys():
                    merged_data[k] = []
                merged_data[k].append(v)
            else:
                k_array = k.split('/')
                k1 = '/'.join([k_array[0], env_name] + k_array[1:])
                merged_data[k1] = v.copy()

        merged_data[f'metadata/dataset{i}'] = env_name
        merged_data[f'metadata/length{i}'] = data_dict['rewards'].shape[0]
        print(f'length of {env_name}: ', data_dict['rewards'].shape[0])

    for k, v in merged_data.items():
        if isinstance(v, list):
            merged_data[k] = np.concatenate(v, axis=0)
        if isinstance(merged_data[k], np.ndarray):
            if k in ['terminals', 'timeouts']:
                dtype = np.bool_
            else:
                dtype = np.float32
            merged_data[k] = merged_data[k].astype(dtype)
        print(k)

    for k in ['observations', 'actions', 'rewards', 'terminals', 'timeouts']:
        print(f'shape of {k}: ', merged_data[k].shape)

    save_filename = os.path.join(DATASET_PATH, gym_env_name + '_' + gym_dataset_name + '-' + gym_env_version + '.hdf5')
    print(f'Starts writing {save_filename}')
    dataset = h5py.File(save_filename, 'w')
    for k in merged_data:
        print(f'type of {k}: ', type(merged_data[k]))
        try:
            dataset.create_dataset(k, data=merged_data[k], compression='gzip')
        except TypeError as e:
            dataset[k] = merged_data[k]
    dataset.close()
    print(f'{save_filename} is saved successfully')
    return save_filename


def test_saved_dataset(filename):
    data_dict = {}
    with h5py.File(filename, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    # Run a few quick sanity checks
    for key in ['observations', 'actions', 'rewards', 'terminals']:
        assert key in data_dict, 'Dataset is missing key %s' % key
    N_samples = data_dict['observations'].shape[0]
    if data_dict['rewards'].shape == (N_samples, 1):
        data_dict['rewards'] = data_dict['rewards'][:, 0]
    assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
        str(data_dict['rewards'].shape))
    if data_dict['terminals'].shape == (N_samples, 1):
        data_dict['terminals'] = data_dict['terminals'][:, 0]
    assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
        str(data_dict['rewards'].shape))

    for k, v in data_dict.items():
        print(k)

    for k in ['observations', 'actions', 'rewards', 'terminals', 'timeouts']:
        print(f'shape of {k}: ', data_dict[k].shape)

    return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_list", default=['walker2d-expert-v2', 'walker2d-medium-v2', 'walker2d-random-v2'])
    args = parser.parse_args()

    saved_filename = merge_dataset(args.env_list)
    import pdb; pdb.set_trace()
    test_saved_dataset(saved_filename)
