from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from collections import defaultdict

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from sklearn.neighbors import NearestNeighbors

import pickle

import os
import numpy as np
import d4rl, gym

def remove_dup(replay_buffer):
    from tqdm import tqdm

    feature = np.concatenate((replay_buffer._observations,
                              replay_buffer._next_obs,
                              replay_buffer._actions,
                              replay_buffer._rewards,
                              replay_buffer._terminals), axis=-1)

    dup_ind_dict = defaultdict(list)
    for i, feat in tqdm(enumerate(feature)):
        dup_ind_dict[tuple(feat)].append(i)

    new_dict = defaultdict(list)
    for key, indices in dup_ind_dict.items():
        first_element = indices[0]
        new_dict[first_element].extend(indices)

    return new_dict

def nearest_neighborhood_cluster(env_name, replay_buffer, n_neighbors=100):

    new_dict = remove_dup(replay_buffer)
    key_index = list(new_dict.keys())
    feature = replay_buffer._observations[key_index]

    file_name = f'./NN_data/nn_data_info' + '_%s' % str(env_name) + '.pickle'

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(
        feature
    )

    distances, idxs = nbrs.kneighbors(feature)
    print("NN done!")

    new_idxs = dict()
    new_dist = dict()

    for key, dup_index in new_dict.items():
        for sub_key in dup_index:
            if sub_key not in idxs[key]:
                new_idxs[sub_key] = np.array(idxs[key])
                new_idxs[sub_key][0] = sub_key
                new_dist[sub_key] = np.array(distances[key])
            else:
                new_idxs[sub_key] = np.array(idxs[key])
                new_dist[sub_key] = np.array(distances[key])

    clustering_info = {}

    sorted_keys = sorted(new_idxs.keys())

    sorted_new_idxs_arrays = np.array([np.array(new_idxs[key]) for key in sorted_keys])
    sorted_new_dist_arrays = np.array([np.array(new_dist[key]) for key in sorted_keys])

    clustering_info['idxs'] = sorted_new_idxs_arrays
    clustering_info['distances'] = sorted_new_dist_arrays

    if not os.path.exists('./NN_data'):
        os.makedirs('./NN_data')

    with open(file_name, 'wb') as fw:
        pickle.dump(clustering_info, fw)
    print("Cluster info saved !: %s" % str(file_name))

    return clustering_info