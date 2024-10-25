import os
import csv
import re
from collections import Iterable

import numpy as np


def get_log_dir_info(base_dir, env_name, arr_algorithm_name):
    p = re.compile('s\d+_\d{6}_\d{6}')

    # Find log dirs and their variant for legend of graph.
    dir_info = []
    for algorithm_name in arr_algorithm_name:
        base_algo_env_dir = os.path.join(base_dir, algorithm_name, env_name)
        print('base_algo_env_dir: ', base_algo_env_dir)
        # print(os.path.isdir(base_algo_env_dir))
        dir_algo_env_info = []
        common_variant = []
        n_variant = 0
        for (root, folder, file) in os.walk(base_algo_env_dir):
            if len(folder) > 0:
                n_run = 0
                dir_run = []
                for fold in folder:
                    # print(fold)
                    m = p.match(fold)
                    if m:
                        prev_dirs = root.split(os.path.sep)
                        hidden = False
                        for pd in prev_dirs:
                            if pd.startswith('_'):
                                hidden = True

                        if not hidden:
                            dir_run.append(fold)
                            n_run += 1
                # print('n_run: ', n_run)
                if n_run > 0:
                    variant = root[len(base_algo_env_dir) + 1:].split(os.path.sep)
                    if n_variant == 0:
                        common_variant = variant
                    else:
                        common_variant = list(set(common_variant) & set(variant))
                    info = {'algorithm_name': algorithm_name, 'log_dir': root, 'run_dir': sorted(dir_run),
                            'n_run': n_run, 'variant': variant}
                    dir_algo_env_info.append(info)
                    n_variant += 1

        # Remove common variant from variants
        # print(f'common_variant : {common_variant}')
        for info in dir_algo_env_info:
            for common_v in common_variant:
                info['variant'].remove(common_v)
        dir_info.extend(dir_algo_env_info)
    return dir_info


def load_progress(dir):
    result = {}
    with open(dir, 'r') as csvfile:
        for i, row in enumerate(csv.DictReader(csvfile)):
            # print(i)
            # print(row)
            # print(row.keys())
            if i == 0:
                for key in row.keys():
                    result[key] = [maychange_str2float(row[key])]
            else:
                for key in row.keys():
                    result[key].append(maychange_str2float(row[key]))

    return result


def rolling_window(x, window_size=5):
    x_t = np.concatenate((np.array(x), np.nan * np.ones(window_size-1)))
    x_t2 = np.nanmean([np.roll(x_t, i) for i in range(window_size)], axis=0)
    return x_t2[:len(x)]



"""
Maybe functions
"""


def maybelist(*args):
    def checknconvert(s):
        if isinstance(s, str):
            return [s]
        elif isinstance(arg, Iterable):
            return s
        else:
            print(f'{arg} is not both str, Iterable')
            return None

    res = []
    if len(args) == 1:
        return checknconvert(args)
    else:
        for arg in args:
            a = checknconvert(arg)
            if a is None:
                return None
            res.append(a)
    return tuple(res)


def maybelistsamelen(target_list, *source_lists):
    def convertlistsamelen(list_t, list_s):
        if not len(list_t) == len(list_s):
            assert len(list_s) == 1, print(f'Length of {list_s} does not match to length of ({list_t})')
            return list_s * len(list_t)
        else:
            return list_s

    if len(source_lists) == 1:
        return convertlistsamelen(list_t=target_list, list_s=source_lists[0])
    else:
        res = []
        for list_s in source_lists:
            res.append(convertlistsamelen(list_t=target_list, list_s=list_s))
        return tuple(res)


def maychange_str2float(str):
    try:
        return float(str)
    except:
        return str




