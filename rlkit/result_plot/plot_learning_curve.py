import os
import re
import csv
from numbers import Number
from reward_info import REF_MAX_SCORE, REF_MIN_SCORE
import numpy as np
import matplotlib

from collections import Iterable
import matplotlib.pyplot as plt

matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode
plt.rcParams['svg.fonttype'] = 'none'

from result_plot.utils import (get_log_dir_info, load_progress, rolling_window,
                               maybelist, maybelistsamelen, maychange_str2float)


def get_performance(performance, timestep_algorithm, max_timesteps=1e5, min_step=1000, normalize_score=False, min_score=0., max_score=1000.):
    performance = np.array(performance)
    max_t = min(timestep_algorithm[-1], max_timesteps)
    performance_t = []
    for i in range(int(max_t / min_step)):
        p = performance[np.array(timestep_algorithm) <= (i+1) * min_step]
        if p.size == 0:
            perf_t = 0
        else:
            perf_t = performance[np.array(timestep_algorithm) <= (i+1) * min_step][-1]
            if normalize_score:
                perf_t = (perf_t - min_score)/ (max_score - min_score) * 100
        performance_t.append(perf_t)
    return np.array(performance_t), (np.arange(int(max_t / min_step)) + 1) * min_step


def plot_learning_curves(base_dir, env_name, arr_algorithm_name, arr_algorithm_label=None,
                         x_name='epoch', x_scale='linear', x_label='Timesteps',
                         y_names=['ReturnAverage'], y_scales='linear', y_labels=None, y_targets=None, compare_axis='x',
                         normalize_score=True, max_timesteps=1e5, with_std=False, with_max_performance=False, with_title=True,
                         fig_size = (6, 4.5), save_filename='Performance', save_fig=True, save_format='pdf',
                         x_step=1e5, min_step=int(1000), plot_separate_seed=False, rolling_window_len=20):
    filename = save_filename
    if with_max_performance:
        filename = "Max_" + filename

    if y_labels is None:
        y_labels = y_names
    if y_targets is None:
        y_targets = ''
    y_names, y_scales, y_labels, y_targets = maybelist(*(y_names, y_scales, y_labels, y_targets))
    y_scales, y_labels, y_targets = maybelistsamelen(y_names, y_scales, y_labels, y_targets)

    max_n_col = 3
    n_y = len(y_names)
    n_row = (n_y - 1) // max_n_col + 1
    n_col = min(n_y, max_n_col)
    if compare_axis == 'y' and n_row == 1:
        n_col = 1
        n_row = n_y

    time_step = (np.arange(0, int(max_timesteps / min_step) + 1) + 1) * min_step
    # Make a structure of plot
    if n_row * n_col == 1:
        fig, axes = plt.subplots(n_row, n_col, figsize=fig_size, sharex=True, sharey=False)
    else:
        fig, axes = plt.subplots(n_row, n_col, sharex=True, sharey=False)

    print(type(axes))
    if not isinstance(axes, Iterable):
        axes = np.array([axes])
    # if isinstance(axes, np.ndarray):
    #     axes = axes.reshape(-1)
    axes = axes.reshape(-1)

    dir_info = get_log_dir_info(base_dir, env_name, arr_algorithm_name)
    if len(dir_info) == 0:
        print("There's no dir to plot")
        raise FileNotFoundError
    print(f'Plotting the following log files: ')
    for dir_i in dir_info:
        print(f'\t{dir_i}')

    max_score = REF_MAX_SCORE[env_name]
    min_score = REF_MIN_SCORE[env_name]
    print(f'max score: {max_score} \t min score: {min_score}')

    color_style_i = 0
    arr_plot = dict()
    arr_name_plot = dict()
    for dir_i in dir_info:
        arr_performance = dict()
        arr_performance_t = dict()
        arr_timestep_t = dict()
        for run_d in dir_i['run_dir']:
            dir = os.path.join(dir_i['log_dir'], run_d, 'progress.csv')
            result = load_progress(dir)
            print('dir: ', dir)
            print(result.keys())

            time_t = (np.array(result['Epoch'], dtype=int) + 1) * 1000
            for y_name in y_names:
                if y_name in result.keys():
                    perf_t = result[y_name]
                elif y_name == 'eval/Average Returns':
                    if 'evaluation/Average_Return' in result.keys():
                        perf_t = result['evaluation/Average Returns']
                    elif 'evaluation/Returns Mean' in result.keys():
                        print('return mean')
                        perf_t = result['evaluation/Returns Mean']
                    else:
                        raise KeyError(f'Both evaluation/Average_Return and evaluation/Returns Mean are not keys of result')
                elif y_name.startswith('eval/'):
                    perf_t = result['evaluation/' + y_name[5:]]
                else:
                    raise KeyError(f'Both {y_name} and evaluation/{y_name[5:]} are not keys of result')

                performance_t, timestep_t = get_performance(perf_t, time_t, max_timesteps=max_timesteps, min_step=min_step,
                                                            normalize_score=normalize_score, min_score=min_score, max_score=max_score)
                print(f'{dir_i["log_dir"]} | performance_t.shape: ', performance_t.shape)
                if y_name not in arr_performance_t.keys():
                    arr_performance_t[y_name] = []
                arr_performance_t[y_name].append(rolling_window(performance_t, window_size=rolling_window_len))

                if y_name not in arr_timestep_t.keys():
                    arr_timestep_t[y_name] = []
                arr_timestep_t[y_name].append(timestep_t)

        if plot_separate_seed:
            for ax, y_name in zip(axes[:len(y_names)], y_names):
                print(y_name)
                color = COLORS[int(color_style_i % len(COLORS))]

                for line_style_i, (timestep_t, performance_t) in enumerate(zip(arr_timestep_t[y_name], arr_performance_t[y_name])):
                    linestyle = LINE_STYLES[int(line_style_i % len(LINE_STYLES))]
                    plot, = ax.plot(timestep_t / x_step, performance_t, color=color, linestyle=linestyle)

                    if y_name not in arr_plot.keys():
                        arr_plot[y_name] = []
                    arr_plot[y_name].append(plot)

                    if y_name not in arr_name_plot.keys():
                        arr_name_plot[y_name] = []
                    if len(dir_i["variant"]) > 0:
                        arr_name_plot[y_name].append(f'{dir_i["algorithm_name"]}_s{line_style_i} ({", ".join(dir_i["variant"])})')
                    else:
                        arr_name_plot[y_name].append(f'{dir_i["algorithm_name"]}_s{line_style_i}')

        else:
            for ax, y_name in zip(axes[:len(y_names)], y_names):
                # Compute minimum timesteps among all runs
                min_len = np.amin([len(timestep_t) for timestep_t in arr_timestep_t[y_name]])
                time_step = arr_timestep_t[y_name][0][:min_len]
                for performance_t in arr_performance_t[y_name]:
                    if y_name not in arr_performance.keys():
                        arr_performance[y_name] = []
                    arr_performance[y_name].append(performance_t[:min_len])

                avg_performance = np.mean(arr_performance[y_name], axis=0)
                std_performance = np.std(arr_performance[y_name], axis=0)
                print(y_name)
                color = COLORS[int(color_style_i % len(COLORS))]
                linestyle = LINE_STYLES[int(color_style_i // len(COLORS))]

                plot, = ax.plot(time_step / x_step, avg_performance, color=color)
                print(f'{y_name} \t| {dir_i["algorithm_name"]} (last timesteps): \t{avg_performance[-1]}')
                print(f'{y_name} \t| {dir_i["algorithm_name"]} (on average): \t{np.mean(avg_performance)}')
                print(f'{y_name} \t| {dir_i["algorithm_name"]} (on average after half): \t{np.mean(avg_performance[int(avg_performance.shape[0] // 2):])}')

                if with_std:
                    upper_performance = avg_performance + std_performance
                    lower_performance = avg_performance - std_performance
                    ax.fill_between(time_step / x_step, lower_performance, upper_performance, color=color, alpha=0.2, linestyle='None')

                if y_name not in arr_plot.keys():
                    arr_plot[y_name] = []
                arr_plot[y_name].append(plot)

                if y_name not in arr_name_plot.keys():
                    arr_name_plot[y_name] = []
                if len(dir_i["variant"]) > 0:
                    arr_name_plot[y_name].append(f'{dir_i["algorithm_name"]} ({", ".join(dir_i["variant"])})')
                else:
                    arr_name_plot[y_name].append(f'{dir_i["algorithm_name"]}')

        color_style_i += 1

    if with_title:
        plt.suptitle(env_name)

    max_legend_plot = None
    max_legend_name_plot = None
    for ax, y_name, y_label, y_scale, y_target in zip(axes[:len(y_names)], y_names, y_labels, y_scales, y_targets):
        if max_legend_plot is None and max_legend_name_plot is None:
            max_legend_plot = arr_plot[y_name]
            max_legend_name_plot = arr_name_plot[y_name]

        if isinstance(y_target, Number):
            straight_line = y_target * np.ones_like(time_step)
            plot, = ax.plot(time_step / x_step, straight_line, color='k', linestyle='--')
            arr_plot[y_name].append(plot)
            arr_name_plot[y_name].append('target')
            max_legend_plot = arr_plot[y_name]
            max_legend_name_plot = arr_name_plot[y_name]

        ax.set_xlim(0, max_timesteps / x_step)
        x_label = f'Training Steps ({x_step:.1e})'
        ax.set_xlabel(x_label)
        ax.set_xscale(x_scale)

        y_label = y_name if y_label is None else y_label
        ax.set_ylabel(y_label)
        ax.set_yscale(y_scale)
        ax.grid(True)

    # axes[0].legend(handles=max_legend_plot, labels=max_legend_name_plot, loc='lower center',
    #                bbox_to_anchor=(0.5, 1.), fancybox=True, shadow=False, ncol=4)
    # axes[0].legend(handles=max_legend_plot, labels=max_legend_name_plot,)
    plt.tight_layout()

    #
    # if arr_name_plot is not None:
    #     plt.legend(list(reversed(arr_plot)), list(reversed(arr_name_plot)))
    # else:
    #     plt.legend(list(reversed(arr_plot)), list(reversed(arr_algorithm_name)))

    # print('---------------------------------------------------------')
    # print(f'{dir_i["algorithm_name"]} ({", ".join(dir_i["variant"])}) : {avg_performance[-1]} +- {std_performance[-1]}')
    # for run_d, performance in zip(dir_i['run_dir'], arr_performance):
    #     print(f'\t {run_d} : {performance[-1]}')


    if save_fig:
        if plot_separate_seed:
            plt.savefig(os.path.join(base_dir, 'figure', filename + "_" + env_name + "_separate." + save_format))
        if with_std:
            plt.savefig(os.path.join(base_dir, 'figure', filename + "_" + env_name + "_fill.pdf"))
        else:
            plt.savefig(os.path.join(base_dir, 'figure', filename + "_" + env_name + "." + save_format))
    plt.show()


# Example usage in jupyter-notebook
# from baselines import log_viewer
# %matplotlib inline
# log_viewer.plot_results(["./log"], 10e6, log_viewer.X_TIMESTEPS, "Breakout")
# Here ./log is a directory containing the monitor.csv files

OBJECTS = ['hopper', 'walker2d', 'halfcheetah', 'ant']
TASKS = ['random', 'medium', 'expert', 'medium-expert', 'medium-replay', 'expert-medium-random']
VERSIONS = [0, 1, 2]
Y_NAMES = [
    'eval/Average Returns', 'trainer/QF Loss', 'trainer/Policy Loss', # 0 ~ 2
    'trainer/QF1 in-distribution values Mean', # 3
    'trainer/QF1 random values Mean', # 4
    'trainer/QF1 next_actions values Mean', # 5
    'trainer/Policy Loss', # 6
    'trainer/Q1 Predictions Mean', # 7
    'trainer/Alpha', 'trainer/Alpha Loss', # 8 ~ 9
    'trainer/Alpha_prime', 'trainer/alpha prime loss', # 10 ~ 11
    'trainer/Log Pis Mean', # 12

]


ENV_INFO = {
    f'{obj}-{task}-v{v}': {'max_timesteps': int(2e6)} for obj in OBJECTS for task in TASKS for v in VERSIONS
}

ALGORITHM_INFO = [
    {'algorithm_name': "OneStepKL", 'name_plot': "OneStep"},  # 0
    {'algorithm_name': "OneStepMinQReg_v1", 'name_plot': "MinQTarg"},  # 1
    {'algorithm_name': "OneStepQReg_regcoef10.0_v0", 'name_plot': "QReg"},  # 2
    {'algorithm_name': "ConservQReg_klreg_v1", 'name_plot': "ConservQReg (v1)"},  # 3


    {'algorithm_name': "BC", 'name_plot': "BC_L2"},  # 4
    {'algorithm_name': "BCPrune_L2_np10_pr0.5_reg_gqa", 'name_plot': "BCPrune_L2_0.5"},  # 5
    {'algorithm_name': "BCPrune_L2_np10_pr0.8_reg_gqa", 'name_plot': "BCPrune_L2_0.8"},  # 6
]


# COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
LINE_STYLES = ['-', '--', '-.', ':', '-']

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data'))

def main():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--object_ind', type=int, default=2) # OBJECTS = ['hopper', 'walker2d', 'halfcheetah', 'ant']
    parser.add_argument('--task_ind', type=int, default=2) # TASKS = ['random', 'medium', 'expert', 'medium-expert', 'medium-replay', 'expert-medium-random']
    parser.add_argument('--env_version', type=int, default=2)
    parser.add_argument('--y_inds', default=[0, 1, 2]) #
    parser.add_argument('--y_scale', help='[linear, log]', default='linear')
    parser.add_argument('--base_dir', help='Base directories', default=BASE_DIR)
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[12, 13, 14, 15, 16, 17])
    parser.add_argument('--algorithm_ind', help='List of algorithms', default=[4, 5, 6])
    parser.add_argument('--fig-size', help='Size of Figure (width, height)', default=(6, 4.5))
    parser.add_argument('--save-fig', type=bool, default=False)
    parser.add_argument('--with-std', type=bool, default=True)
    parser.add_argument('--with-max-performance', type=bool, default=True)
    parser.add_argument('--with-title', type=bool, default=False)
    parser.add_argument('--with-all-actors', type=bool, default=False)
    parser.add_argument('--save-format', help='figure format (eps, fig, png, etc)', default="eps")

    args = parser.parse_args()
    args.algorithm_name = [ALGORITHM_INFO[algorithm_ind]['algorithm_name'] for algorithm_ind in args.algorithm_ind]
    args.name_plot = [ALGORITHM_INFO[algorithm_ind]['name_plot'] for algorithm_ind in args.algorithm_ind]
    args.env_name = f'{OBJECTS[args.object_ind]}-{TASKS[args.task_ind]}-v{args.env_version}'
    args.max_timesteps = ENV_INFO[args.env_name]['max_timesteps']
    args.y_names = [Y_NAMES[i] for i in args.y_inds]
    plot_learning_curves(args.base_dir, args.env_name, args.algorithm_name, y_names=args.y_names, y_scales=args.y_scale,
                         normalize_score=True, max_timesteps=args.max_timesteps, with_std=args.with_std,
                         with_max_performance=args.with_max_performance, with_title=args.with_title,
                         fig_size=args.fig_size, save_filename='Performance', save_fig=args.save_fig,
                         save_format=args.save_format, plot_separate_seed=False, rolling_window_len=10)


if __name__ == '__main__':
    main()
