"""
Trainer for QM9 dataset supporting mixture policy training with many models.
"""

import os
import socket
import shutil
from copy import copy

import torch
from gflownet.config import Config
from gflownet.trainer import RewardScalar
from gflownet.mixture_trainer import MixtureOnlineTrainer

from gflownet.data.qm9 import QM9Dataset
from gflownet.tasks.qm9.qm9 import QM9GapTask
from gflownet.envs.mol_building_env import MolBuildingEnvContext


class QM9MixtureModelTrainer(MixtureOnlineTrainer):
    task: QM9GapTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = True
        
        cfg.num_workers = 8
        cfg.checkpoint_every = 1000
        cfg.num_training_steps = 100000
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.global_batch_size = 64
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4
        cfg.cond.temperature.sample_dist = "uniform"
        cfg.cond.temperature.dist_params = [0.5, 32.0]
        cfg.cond.temperature.num_thermometer_dim = 32

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.0
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.001
        cfg.algo.rl_train_random_action_prob = 0.001
        cfg.algo.alpha = 1.0
        cfg.algo.p_greedy_sample = False
        cfg.algo.p_of_max_sample = False
        cfg.algo.p_quantile_sample = False
        cfg.algo.scheduler_type='cosine_annealing'
        cfg.algo.scheduler_step=1500
        cfg.algo.p = 0.99
        cfg.algo.ddqn_update_step = 1
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_offline_ratio = 0
        cfg.algo.offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False

        cfg.replay.use = False
        cfg.replay.capacity = 5_000
        cfg.replay.warmup = 1_000
    
    def setup_task(self):
        self.task = QM9GapTask(
            dataset=self.training_data,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )
        self.second_task = copy(self.task)
        # Ignore temperature for RL task 
        self.second_task.cond_info_to_logreward = lambda cond_info, flat_reward: RewardScalar(
            flat_reward.reshape((-1,))
        )

    def setup_data(self):
        self.training_data = QM9Dataset(self.cfg.task.qm9.h5_path, train=True, target="gap")
        self.test_data = QM9Dataset(self.cfg.task.qm9.h5_path, train=False, target="gap")

    def setup_env_context(self):
        self.ctx = MolBuildingEnvContext(
            ["C", "N", "F", "O"],
            expl_H_range=[0, 1, 2, 3],
            num_cond_dim=self.task.num_cond_dim,
            allow_5_valence_nitrogen=True
        )
        # Note: we only need the allow_5_valence_nitrogen flag because of how we generate trajectories
        # from the dataset. For example, consider tue Nitrogen atom in this: C[NH+](C)C, when s=CN(C)C, if the action
        # for setting the explicit hydrogen is used before the positive charge is set, it will be considered
        # an invalid action. However, generate_forward_trajectory does not consider this implementation detail,
        # it assumes that attribute-setting will always be valid. For the molecular environment, as of writing
        # (PR #98) this edge case is the only case where the ordering in which attributes are set can matter.


def main():
    """Example of how this model can be run outside of Determined"""
    hps: Config = {
        "log_dir": "cd",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "overwrite_existing_exp": True,
        "num_training_steps": 100000,
        "validate_every": 0,
        "num_workers": 1,
        "opt": {
            "lr_decay": 20000
        },
        "algo": {
            "p_greedy_sample": False,
            "p_of_max_sample": False,
            "p_quantile_sample": False,
            "p": 0.99,
            "dqn_n_step": 3,
            "sampling_tau": 0.0,
            "global_batch_size": 64,
            "ddqn_update_step": 1,
            "rl_train_random_action_prob": 0.1,
            "dqn_tau": 0.9
        },
        "cond": {
            "temperature": {
                "sample_dist": "uniform", 
                "dist_params": [0.5, 32.0]
            },
        },
        "task": {
            "qm9": {
                "h5_path": "./data/qm9.h5",
                "model_path": "./data/mxmnet_gap_model.pt"
            }
        }
    }

    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = QM9MixtureModelTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()