import os
import shutil
import socket
import pickle
from copy import copy, deepcopy

from gflownet.config import Config
from gflownet.trainer import RewardScalar
from gflownet.envs.seq_building_env import (
    AutoregressiveSeqBuildingContext,
    PrependAppendSeqBuildingContext,
    SeqBuildingEnv
)
from gflownet.tasks.bitseq.bitseq import BitSeqTask
from gflownet.models.seq_transformer import SeqTransformerGFN
from gflownet.mixture_trainer import MixtureOnlineTrainer


class BitSeqMixTrainer(MixtureOnlineTrainer):
    task: BitSeqTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = True
        cfg.num_workers = 8
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.offline_ratio = 0

        # Hyperparameters for the transformer sampling model same as (Malkin et al., 2023)
        cfg.model.num_emb = 64
        cfg.model.num_layers = 3
        cfg.model.seq_transformer.num_heads = 8

        # Trajectory length hyperparameters. We want to generate sequences of 
        # n=120 bits (30 nodes * k=4 bits per node) so this is set to n / k
        cfg.algo.max_nodes = 30
        cfg.algo.max_edges = 30
        cfg.algo.max_len = 30
        cfg.algo.min_len = 30

        # Other hyperparameter choices in (Malkin et al., 2023)
        cfg.num_training_steps = 50_000
        cfg.algo.global_batch_size = 16
        cfg.algo.train_random_action_prob = 5e-4
        cfg.cond.temperature.sample_dist = "constant"
        cfg.cond.temperature.dist_params = [3.0]
        cfg.cond.temperature.num_thermometer_dim = 1
        cfg.opt.learning_rate = 1e-4
        cfg.algo.tb.Z_learning_rate = 1e-3

        # QGFN hyperparameters
        cfg.algo.method = "TB"
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.rl_train_random_action_prob = 5e-4
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.p_greedy_sample = False
        cfg.algo.p_of_max_sample = False
        cfg.algo.p_quantile_sample = False
        cfg.algo.p = 0.99
        cfg.algo.ddqn_update_step = 1
        cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False

        # Replay buffer hyperparameters
        cfg.replay.use = False
        cfg.replay.capacity = 5_000
        cfg.replay.warmup = 1_000

    def setup_model(self):
        # NOTE: do we care about shared layers?
        self.model = SeqTransformerGFN(
            self.ctx,
            self.cfg,
            min_len=self.cfg.algo.max_len,
            variant=self.cfg.task.bitseq.variant,
        )
        self.second_model = SeqTransformerGFN(
            self.second_ctx,
            self.cfg,
            min_len=self.cfg.algo.max_len,
            variant=self.cfg.task.bitseq.variant,
        )

        self._get_additional_parameters = lambda: list(self.second_model.parameters())
        self.second_model_lagged = deepcopy(self.second_model)
        self.second_model_lagged.to(self.device)
        self.dqn_tau = self.cfg.algo.dqn_tau
        self.ddqn_update_step = self.cfg.algo.ddqn_update_step

    def setup_task(self):
        # Check that the task variant is valid
        assert self.cfg.task.bitseq.variant in ["autoregressive", "prepend-append"], (
            f"task variant must be one of ['autoregressive', 'prepend-append'], got {self.cfg.task.bitseq.variant}"
        )

        # Load the target sequences from the pickle file
        with open(self.cfg.task.bitseq.modes_path, "rb") as f:
            modes = pickle.load(f)

        # Check that all target sequences have the same specified length
        num_bits = self.cfg.task.bitseq.k * self.cfg.algo.max_nodes
        assert all([len(m) == num_bits for m in modes]), (
            f"all target sequences must have the same length {self.cfg.algo.max_nodes}"
        )

        self.task = BitSeqTask(
            seqs=modes,
            cfg=self.cfg,
            rng=self.rng,
        )
        self.second_task = copy(self.task)
        self.second_task.cond_info_to_logreward = \
            lambda _, flat_reward: RewardScalar(flat_reward.reshape((-1,)))

    def setup_env_context(self):
        self.env = SeqBuildingEnv(variant=self.cfg.task.bitseq.variant)
        if self.cfg.task.bitseq.variant == "prepend-append":
            self.ctx = PrependAppendSeqBuildingContext(
                alphabet=self.task.vocab,
                num_cond_dim=self.task.num_cond_dim,
                min_len=self.cfg.algo.max_len,
            )
        else:
            self.ctx = AutoregressiveSeqBuildingContext(
                alphabet=self.task.vocab,
                num_cond_dim=self.task.num_cond_dim,
                min_len=self.cfg.algo.max_len,
            )

    def setup_algo(self):
        super().setup_algo()
        # If the algo implements it, avoid giving, ["A", "AB", "ABC", ...] as a sequence of inputs, and instead give
        # "ABC...Z" as a single input, but grab the logits at every timestep. Only works if using a transformer with
        # causal self-attention.
        if self.cfg.task.bitseq.variant == "autoregressive":
            self.algo.model_is_autoregressive = True
        else:
            self.algo.model_is_autoregressive = False


def main():
    """Example of how this model can be run outside of Determined"""
    hps = {
        "log_dir": "/home/stephzlu/scratch/bitseq_logs",
        "device": "cpu",
        "overwrite_existing_exp": True,
        "num_training_steps": 50_000,
        "checkpoint_every": 1000,
        "validate_every": 0,
        "num_workers": 1,
        "cond": {
            "temperature": {
                "sample_dist": "constant",
                "dist_params": [3.0],
                "num_thermometer_dim": 1,
            }
        },
        "algo": {
            "train_random_action_prob": 5e-4,
            "p_greedy_sample": True,
            "high_Q_sample": False,
            "p": 0.99,
            "dqn_n_step": 30,   # all trajectories have length 30
            "sampling_tau": 0.9,
            "global_batch_size": 16,
            "sampling_ratio": 1.0,
            "ddqn_update_step": 1,
            "rl_train_random_action_prob": 5e-4,
            "dqn_tau": 0.9,
            "tb": { "variant": "TB" },
        },
        "task": {
            "bitseq": {
                "variant": "prepend-append",
                "modes_path": "./data/modes.pkl",
                "k": 4,
            }
        }
    }
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = BitSeqMixTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
