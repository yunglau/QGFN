"""
This is a generative script that will generate the executables and log directories 
for a set of hyperparameters provided to a GFlowNet trainer object. To use this script,
simply modify the hyperparameters in the main function below and run the script.
"""

import os
from itertools import product
from gflownet.config import Config
from utils.runs import RunObject

TASK = 'seh'
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
LOG_ROOT = f'{CUR_DIR}/logs'

BASE_HPS: Config = {
    "log_dir": '',
    "device": 'cuda',
    "overwrite_existing_exp": True,
    "num_training_steps": 10000,
    "validate_every": 0,
    "num_workers": 0,
    "opt": {
        "lr_decay": 2000,
    },
    "cond": {
        "temperature": {
            "sample_dist": "constant",
            "dist_params": 32.0,
        }
    },
    "algo": {
        "p_greedy_sample": False,
        "p_of_max_sample": False,
        "p_quantile_sample": False,
        "p": 0.0,
        "dqn_n_step": 25,   # all trajectories have length 30
        "sampling_tau": 0.99,
        "global_batch_size": 16,
        "ddqn_update_step": 1,
        "train_random_action_prob": 0.01,
        "rl_train_random_action_prob": 0.1,
        "dqn_tau": 0.9,
        "tb": { "variant": "TB" },
    },
    "task": {
        "qm9": {
            "h5_path": "path.to.dataset/qm9.h5",
            "model_path": "path.to.model/mxmnet_gap_model.pt"
        },
        "bitseq": {
            "variant": "prepend-append",
            "modes_path": "path.to.ref.sequences/modes.pkl",
            "k": 4,
        }
    }
}


if __name__ == '__main__':
    # Define a list of hyperparameters to test
    temperature_values: [[float]] = [32.0]
    replay_values: [bool] = [False]
    tb_values: [str] = ["SubTB1"]
    # sampling_ratios: [float] = [1.0]
    ddqn_update_steps: [int] = [1]
    replay_buffer_size: [int] = [10_000]
    dqn_taus: [float] = [0.95]
    dqn_epsilons: [float] = [0.10]
    num_workers: [int] = [8]
    num_training_steps: [int] = [10_000]
    batch_size: [int] = [64]
    p_greedy_sample: [bool] = [True, False]
    p_of_max_sample: [bool] = [True, False]
    p_quantile_sample: [bool] = [True, False]
    prob: [float] = [0.4, 0.5, 0.7, 0.8, 0.9, 0.91]
    dqn_n_step: [float] = [25]

    # Define the specific configurations
    specific_configs = [
        {'p_greedy_sample': True, 'p_of_max_sample': False, 'p_quantile_sample': False, 'prob': 0.4},
        {'p_greedy_sample': True, 'p_of_max_sample': False, 'p_quantile_sample': False, 'prob': 0.5},
        {'p_greedy_sample': False, 'p_of_max_sample': True, 'p_quantile_sample': False, 'prob': 0.9},
        {'p_greedy_sample': False, 'p_of_max_sample': True, 'p_quantile_sample': False, 'prob': 0.91},
        {'p_greedy_sample': False, 'p_of_max_sample': False, 'p_quantile_sample': True, 'prob': 0.7},
        {'p_greedy_sample': False, 'p_of_max_sample': False, 'p_quantile_sample': True, 'prob': 0.8}
    ]

    # Create scripts for each specific configuration
    params = product(temperature_values, replay_values, tb_values, 
                     ddqn_update_steps, replay_buffer_size, dqn_taus, dqn_epsilons, num_workers, 
                     num_training_steps, batch_size, p_greedy_sample, p_of_max_sample, p_quantile_sample, prob, dqn_n_step)

    for p in params:
        (temperature, replay, tb, ddqn_update_step, replay_buffer, dqn_tau, dqn_epsilon, 
         num_worker, num_training_step, batch, p_greedy_sample, p_of_max_sample, p_quantile_sample, prob, dqn_step) = p
        
        # Check if the current combination matches any specific configuration
        match_found = False
        for config in specific_configs:
            if (config['p_greedy_sample'] == p_greedy_sample and
                config['p_of_max_sample'] == p_of_max_sample and
                config['p_quantile_sample'] == p_quantile_sample and
                config['prob'] == prob):
                match_found = True
                break
        
        if match_found:
            BASE_HPS['algo']['p_greedy_sample'] = p_greedy_sample
            BASE_HPS['algo']['p_of_max_sample'] = p_of_max_sample
            BASE_HPS['algo']['p_quantile_sample'] = p_quantile_sample
            BASE_HPS['algo']['p'] = prob

            run_obj = RunObject(task=TASK, p=p, LOG_ROOT=LOG_ROOT)
            run_obj.print_obj()
            run_obj.generate_scripts(CUR_DIR=CUR_DIR, BASE_HPS=BASE_HPS)