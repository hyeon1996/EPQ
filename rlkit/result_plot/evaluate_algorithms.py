import os
import re
import datetime
import dateutil.tz
from copy import deepcopy

import numpy as np
import torch
import gym, d4rl
from tqdm import tqdm
# import pandas as pd
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.rollout_functions import rollout
from result_plot.utils import get_log_dir_info
from reward_info import REF_MAX_SCORE, REF_MIN_SCORE


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DATADISK_DIR = os.path.join("/media", "datadisk", "cql_data")

ABBR = {
    'expert': 'Exp',
    'medium': 'Med',
    'random': 'Rnd',
    'medium-expert': 'MedExp',
    'medium-replay': 'MedRep',
}

FULL = {
    'Exp': 'expert',
    'Med': 'medium',
    'Rnd': 'random',
    'MedExp': 'medium-expert',
    'MedRep': 'medium-replay',
}



def eval_algo(algo_name, env_name, last_itr=None, n_eval_traj=100, max_path_length=1000):
    base_model_dir = DATADISK_DIR if os.path.isdir(DATADISK_DIR) else BASE_DIR
    base_eval_dir = os.path.join(os.path.dirname(__file__), "evaluations")

    # Get dir_info
    log_dir_info = get_log_dir_info(base_model_dir, env_name=env_name, arr_algorithm_name=[algo_name])
    if len(log_dir_info) == 0:
        print("There is no dir to plot")
        raise FileNotFoundError
    print(f"Evaluating the following log files: ")
    for dir_i in log_dir_info:
        print(f"\t{dir_i}")

    dir_i = log_dir_info[0]
    model_dir = dir_i["log_dir"]
    run_dir_need_eval = deepcopy(dir_i["run_dir"])

    # Get evaluated_dir info and compare it to dir_info
    evaluation_file = os.path.join(base_eval_dir, f"{env_name}_{algo_name}.npz")
    if os.path.isfile(evaluation_file):
        evaluation = load_evaluation(evaluation_file, print_info=True)

        for evaluated_dir in evaluation["evaluated_dir"]:
            run_dir_need_eval.remove(evaluated_dir)
    else:
        evaluation = {"lengths": None, "scores": None, "normalized_scores": None, "evaluated_dir": None}

    if len(run_dir_need_eval) == 0:
        print("All run dirs are already evaluated!")
    else:
        print("Run dirs that needs to evaluate are")
        for run_dir in run_dir_need_eval:
            print(f"\t{os.path.join(model_dir, run_dir)}")

        min_score, max_score = REF_MIN_SCORE[env_name], REF_MAX_SCORE[env_name]

        for run_dir in run_dir_need_eval:
            if last_itr and not os.path.isfile(os.path.join(model_dir, run_dir, f"itr_{last_itr}.pkl")):
                print(f"{model_dir}/{run_dir}/params.pkl is passed since this is currently running!")
                continue

            model_file = os.path.join(model_dir, run_dir, f"params.pkl")
            if not os.path.isfile(model_file):
                print(f"{model_file} does not exists")
                raise FileNotFoundError

            snapshot = torch.load(model_file)
            policy = snapshot["trainer/policy"]
            env = gym.make(env_name)
            lengths, scores, normalized_scores = [], [], []

            eval_policy = MakeDeterministic(policy).to("cpu")
            for _ in tqdm(range(n_eval_traj), desc=f"{model_dir}/{run_dir}"):
                path = rollout(
                    env,
                    eval_policy,
                    max_path_length=max_path_length,
                )
                length = len(path["rewards"])
                score = np.sum(path["rewards"])
                normalized_score = (score - min_score)/ (max_score - min_score) * 100.

                lengths.append(length)
                scores.append(score)
                normalized_scores.append(normalized_score)

            if evaluation["lengths"] is None:
                evaluation["lengths"] = [lengths]
                evaluation["scores"] = [scores]
                evaluation["normalized_scores"] = [normalized_scores]
                evaluation["evaluated_dir"] = [run_dir]
            else:
                evaluation["lengths"].append(lengths)
                evaluation["scores"].append(scores)
                evaluation["normalized_scores"].append(normalized_scores)
                evaluation["evaluated_dir"].append(run_dir)

        save_evaluation(evaluation_file, print_info=True, **evaluation)


def load_evaluation(file, print_info=False):
    if not os.path.isfile(file):
        print(f"{file} does not exists")
        raise FileNotFoundError
    evaluation = dict(np.load(file))
    for k, v in evaluation.items():
        evaluation[k] = v.tolist()

    if print_info:
        print(f"{file} is loaded successfully!")
    return evaluation


def save_evaluation(file, scores, normalized_scores, print_info=False, **kwargs):
    np.savez(file, scores=scores, normalized_scores=normalized_scores, **kwargs)
    if print_info:
        print(f"{file} is saved successfully!")


class Evaluation_Recorder(object):
    def __init__(self, base_eval_dir=None, base_model_dir=None, n_eval_traj=100):
        self.base_eval_dir = base_eval_dir if base_eval_dir else os.path.join(os.path.dirname(__file__), "evaluations")
        if base_model_dir:
            self.base_model_dir = base_model_dir
        else:
            self.base_model_dir = DATADISK_DIR if os.path.isdir(DATADISK_DIR) else BASE_DIR

        if not os.path.isdir(self.base_eval_dir):
            os.makedirs(self.base_eval_dir)

        self.task = ["expert", "medium", "random", "medium-expert", "medium-replay"]
        self.robot = ["hopper", "walker2d", "halfcheetah"]
        self.env_v = 2

        self.n_eval_traj = n_eval_traj
        self.possible_print_modes = ["human", "latex"]

        self.initialize()
        self.update_algo_evaluations(self.evaluated_algo, reevaluate_if_exist=False)

    def initialize(self):
        self.envs = []
        for task in self.task:
            for robot in self.robot:
                self.envs.append(f"{robot}-{task}")

        self.evaluated_algo = []
        files = os.listdir(self.base_eval_dir)
        for file in files:
            file_t = file[:-4].split("_") # Remove file extenstion .npz then spilit
            if len(file_t) == 1:
                continue
            env_name = file_t[0][:-3] # Get env_name
            if env_name not in self.envs:
                print(f"{env_name} is passed!")
                continue
            algo_name = "_".join(file_t[1:]) # Get algo_name
            if algo_name not in self.evaluated_algo:
                self.evaluated_algo.append(algo_name)

        self.evaluation_board = {}
        for env_name in self.envs:
            self.evaluation_board[env_name] = {}
            for algo_name in self.evaluated_algo:
                evaluation_file = os.path.join(self.base_eval_dir, f"{env_name}-v{self.env_v}_{algo_name}.npz")
                if os.path.isfile(evaluation_file):
                    evaluation = load_evaluation(evaluation_file)
                    evaluation_algo = {
                        "lengths": evaluation["lengths"], "scores": evaluation["scores"],
                        "normalized_scores": evaluation["normalized_scores"],
                        "evaluated_dir": evaluation["evaluated_dir"]
                    }
                    self.evaluation_board[env_name][algo_name] = evaluation_algo
                else:
                    self.evaluation_board[env_name][algo_name] = {
                        "lengths": None, "scores": None, "normalized_scores": None, "evaluated_dir": None
                    }
        print("initialization is successfully done!")

    def update_algo_evaluations(self, algo_names, reevaluate_if_exist=False):
        for env_name in self.envs:
            for algo_name in algo_names:
                self.update_evaluation(algo_name=algo_name, env_name=env_name+f"-v{self.env_v}", reevaluate_if_exist=reevaluate_if_exist)
        self.initialize()

    def update_evaluation(self, algo_name, env_name, last_itr=None, max_path_length=1000, reevaluate_if_exist=False):
        # Get dir_info
        log_dir_info = get_log_dir_info(self.base_model_dir, env_name=env_name, arr_algorithm_name=[algo_name])
        if len(log_dir_info) == 0:
            print(f"There is no dir to evaluate in {self.base_model_dir}/{algo_name}/{env_name}")
            return
        # print(f"Evaluating the following log files: ")
        # for dir_i in log_dir_info:
        #     print(f"\t{dir_i}")

        dir_i = log_dir_info[0]
        model_dir = dir_i["log_dir"]
        run_dir_need_eval = deepcopy(dir_i["run_dir"])

        # Get evaluated_dir info and compare it to dir_info
        evaluation_file = os.path.join(self.base_eval_dir, f"{env_name}_{algo_name}.npz")
        print('reevaluate_if_exist: ', reevaluate_if_exist)
        if os.path.isfile(evaluation_file) and not reevaluate_if_exist:
            evaluation = load_evaluation(evaluation_file, print_info=True)

            for evaluated_dir in evaluation["evaluated_dir"]:
                if evaluated_dir in run_dir_need_eval:
                    run_dir_need_eval.remove(evaluated_dir)
        else:
            evaluation = {"lengths": None, "scores": None, "normalized_scores": None, "evaluated_dir": None}

        if len(run_dir_need_eval) == 0:
            print(f"All run dirs in {model_dir} are already evaluated!")
        else:
            print("Run dirs that need to evaluate are")
            for run_dir in run_dir_need_eval:
                print(f"\t{os.path.join(model_dir, run_dir)}")

            min_score, max_score = REF_MIN_SCORE[env_name], REF_MAX_SCORE[env_name]

            for run_dir in run_dir_need_eval:
                if last_itr and not os.path.isfile(os.path.join(model_dir, run_dir, f"itr_{last_itr}.pkl")):
                    print(f"{model_dir}/{run_dir}/params.pkl is passed since this is currently running!")
                    continue

                model_file = os.path.join(model_dir, run_dir, f"params.pkl")
                if not os.path.isfile(model_file):
                    print(f"{model_file} does not exists")
                    raise FileNotFoundError

                snapshot = torch.load(model_file)
                policy = snapshot["trainer/policy"]
                env = gym.make(env_name)
                lengths, scores, normalized_scores = [], [], []

                eval_policy = MakeDeterministic(policy).to("cpu")
                for _ in tqdm(range(self.n_eval_traj), desc=f"{model_dir}/{run_dir}"):
                    path = rollout(
                        env,
                        eval_policy,
                        max_path_length=max_path_length,
                    )
                    length = len(path["rewards"])
                    score = np.sum(path["rewards"])
                    normalized_score = (score - min_score) / (max_score - min_score) * 100.

                    lengths.append(length)
                    scores.append(score)
                    normalized_scores.append(normalized_score)

                if evaluation["lengths"] is None:
                    evaluation["lengths"] = [lengths]
                    evaluation["scores"] = [scores]
                    evaluation["normalized_scores"] = [normalized_scores]
                    evaluation["evaluated_dir"] = [run_dir]
                else:
                    evaluation["lengths"].append(lengths)
                    evaluation["scores"].append(scores)
                    evaluation["normalized_scores"].append(normalized_scores)
                    evaluation["evaluated_dir"].append(run_dir)

            save_evaluation(evaluation_file, print_info=True, **evaluation)

    def print_evaluation_table(self, tasks=[], robots=[], algos=[], base_dir=None, modes=["human", "latex"], with_std=False, transpose=False):
        table_base_dir = base_dir if base_dir else os.path.join(os.path.dirname(__file__), "eval_table")
        if not os.path.isdir(table_base_dir):
            os.makedirs(table_base_dir)

        # Check modes in self.possible_print_modes
        for mode in modes:
            assert mode in self.possible_print_modes, f"{mode} is not in possible_print_modes ({self.possible_print_modes})"

        tasks = tasks if len(tasks) else self.task
        robots = robots if len(robots) else self.robot
        print(len(algos))
        print(algos)
        algos = algos if len(algos) else self.evaluated_algo


        # Change "all" to corresponding names
        env_keys = []
        eval_dict = dict()
        n_row = len(tasks) * len(robots)
        row_i = 0
        for task in tasks:
            for robot in robots:
                env_keys.append(f"{task}/{robot}")
                for algo_name in algos:
                    eval_env_algo = self.evaluation_board[f"{robot}-{task}"][algo_name]
                    for k, v in eval_env_algo.items():
                        if k == "evaluated_dir":
                            if "summary" not in eval_dict.keys():
                                eval_dict["summary"] = [[] for _ in range(n_row)]
                            if v is None:
                                eval_dict["summary"][row_i].append("0")
                            else:
                                eval_dict["summary"][row_i].append(f"{len(v)}")
                        else:
                            if k not in eval_dict.keys():
                                eval_dict[k] = [[] for _ in range(n_row)]

                            if v is None:
                                eval_dict[k][row_i].append("-")
                                continue
                            v_t = np.mean(v, axis=1)
                            v_mean = np.round(np.mean(v_t) * 10.) / 10.
                            v_std = np.round(np.std(v_t) * 10.) / 10.
                            if with_std:
                                eval_dict[k][row_i].append(f"{v_mean:.1f} +- {v_std:.1f}")
                            else:
                                eval_dict[k][row_i].append(f"{v_mean:.1f}")
                row_i += 1

        for k, v in eval_dict.items():
            if transpose:
                evaluation_table_filename = os.path.join(table_base_dir, f"Table_{k}_trans")
                v1 = np.array(v).T.tolist()

                first_row, second_row = [" "], [" "]
                for env_key in env_keys:
                    env_k = env_key.split("/")
                    first_row.append(env_k[0])
                    second_row.append(env_k[1])

                print("=====================================================")
                for mode in modes:
                    if mode == "human":
                        with open(f"{evaluation_table_filename}.csv", "w") as f:
                            f.write(",".join(first_row) + "\n")
                            f.write(",".join(second_row) + "\n")
                            for i in range(len(v1)):
                                f.write(",".join([algos[i]] + v1[i]))
                                if i < len(v1)-1:
                                    f.write("\n")
                        print(f"{evaluation_table_filename}.csv is saved successfully!")
                    elif mode == "latex" and "scores" in k:
                        with open(f"{evaluation_table_filename}.txt", "w") as f:
                            f.write("\\begin{table}[!h]\n")
                            f.write("\\centering\n")
                            f.write("\\caption{} \\label{table:} \n")
                            f.write("\\begin{tabular}{ |" + "|".join(["c"]*len(first_row)) + "| } \n")
                            indent = "  "
                            next_line = "\\\\ \n"
                            f.write(indent + "\\hline \n")
                            merged_first_row = [first_row[0]]
                            for i, el in enumerate(first_row):
                                if (i - 1) % len(robots) == 0:
                                    v2 = "\\multicolumn{" + str(len(robots)) + "}{c|}{" + el + "}"
                                    merged_first_row.append(v2)

                            f.write(indent + " &".join(merged_first_row) + next_line)
                            f.write(indent + "\\hline \n")
                            f.write(indent + " &".join(second_row) + next_line)
                            f.write(indent + "\\hline \n")
                            for i in range(len(v1)):
                                f.write(indent + " &".join([algos[i].replace("_", "\\_")] + v1[i]) + next_line)
                            f.write(indent + "\\hline \n")
                            f.write("\\end{tabular} \n")
                            f.write("\\end{table}")
                        print(f"{evaluation_table_filename}.txt is saved successfully!")

            else:
                evaluation_table_filename = os.path.join(table_base_dir, f"Table_{k}")
                v1 = np.copy(v).tolist()

                first_column, second_column = [], []
                for env_key in env_keys:
                    env_k = env_key.split("/")
                    first_column.append(env_k[0])
                    second_column.append(env_k[1])

                for mode in modes:
                    if mode == "human":
                        with open(f"{evaluation_table_filename}.csv", "w") as f:
                            f.write(",".join([" ", " "] + algos) + "\n")
                            for i in range(len(v1)):
                                f.write(",".join([first_column[i], second_column[i]] + v1[i]))
                                if i < len(v1) - 1:
                                    f.write("\n")
                        print(f"{evaluation_table_filename}.csv is saved successfully!")
                    elif mode == "latex" and "scores" in k:
                        with open(f"{evaluation_table_filename}.txt", "w") as f:
                            f.write("\\begin{table}[!h]\n")
                            f.write("\\centering\n")
                            f.write("\\caption{} \\label{table:} \n")
                            f.write("\\begin{tabular}{ |" + "|".join(["c"] * int(len(algos)+2)) + "| } \n")
                            indent = "  "
                            next_line = "\\\\ \n"
                            f.write(indent + "\\hline \n")
                            f.write(indent + " &".join([" ", " "] + [algo.replace("_", "\\_") for algo in algos]) + next_line)
                            f.write(indent + "\\hline \n")
                            for i in range(len(v1)):
                                if i % len(robots) == 0:
                                    v2 = "\\multirow{" + str(len(robots)) + "}{*}{\\rotatebox[origin=c]{90}{" + ABBR[first_column[i]] + "}}"
                                else:
                                    v2 = " "
                                f.write(indent + " &".join([v2, second_column[i]] + v1[i]) + next_line)
                                if (i+1) % len(robots) == 0:
                                    f.write(indent + "\\hline \n")

                            f.write("\\end{tabular} \n")
                            f.write("\\end{table}")
                        print(f"{evaluation_table_filename}.txt is saved successfully!")


if __name__ == "__main__":
    # eval_algo("BC", "walker2d-medium-v2", last_itr=2000, n_eval_traj=100, max_path_length=1000)
    algos = ["BC", "CQL_new", "OneStepKL", "MinQTargKL_v1", "OneStepQReg_klreg_regcoef10.0_v0", "OneStepQReg_regcoef10.0_v0",
             "ConservQReg_klreg_v1", #"ConservQReg_klreg_regcoef10.0_v2", "ConservQReg_klreg_regcoef20.0_v2"
             ]
    # algos = ["MinQTargKL_v1", "OneStepQReg_klreg_regcoef10.0_v0", "ConservQReg_klreg_v1"]
    recorder = Evaluation_Recorder()
    recorder.update_algo_evaluations(algos, reevaluate_if_exist=False)
    recorder.print_evaluation_table(algos=algos, with_std=False, transpose=False)








