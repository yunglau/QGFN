import os
import shutil
import socket
from typing import Dict, List, Tuple

import pickle
import numpy as np
import torch
from torch import Tensor
import editdistance

from gflownet.config import Config
from gflownet.envs.seq_building_env import (
    AutoregressiveSeqBuildingContext,
    PrependAppendSeqBuildingContext,
    SeqBuildingEnv
)
from gflownet.models.seq_transformer import SeqTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.utils.conditioning import TemperatureConditional


class BitSeqTask(GFNTask):
    """Sets up a task where the reward is inversely proportional to the smallest edit distance between
    the generated sequence and the nearest sequence in a predefined set. See Malkin et al. (2023).
    Additionally, we support two task variants: autoregressive or prepend-append"""

    def __init__(
        self,
        seqs: List[str],            # List of target sequences to compare against (modes)
        cfg: Config,
        rng: np.random.Generator,
    ):
        self.seqs = seqs                                # list of target sequences
        self.k = cfg.task.bitseq.k                      # number of bits to append or prepend in each step (def: 4)
        self.norm = cfg.algo.max_nodes*self.k           # length of each sequence in bits (def: 120)

        # vocab is all possible bit sequences of length k
        self.vocab = ["".join(map(str, x)) for x in np.ndindex((2,) * self.k)]

        assert len(self.vocab) == 2 ** self.k, "vocab size must be 2^k"

        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.temperature_conditional.transform(cond_info, flat_reward))

    def compute_flat_rewards(self, objs: List[str]) -> Tuple[FlatRewards, Tensor]:
        """R(x) = exp(1 − min_{y \in M} d(x, y)/n)"""
        # assert all([len(o) == self.norm for o in objs]), f"all objects must have length {self.norm}"
        ds = torch.tensor([min([editdistance.eval(s, p) for p in self.seqs]) for s in objs]).float()
        rs = torch.exp(1-(ds/self.norm))
        return FlatRewards(rs[:, None]), torch.ones(len(objs), dtype=torch.bool)


class BitSeqTrainer(StandardOnlineTrainer):
    task: BitSeqTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
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
        cfg.algo.scale_temp = False

        cfg.algo.method = "TB"
        cfg.algo.sampling_tau = 0.9
        cfg.algo.sampling_ratio = 1.0
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.p_greedy_sample = False
        cfg.algo.high_Q_sample = False
        cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False

    def setup_model(self):
        self.model = SeqTransformerGFN(
            self.ctx,
            self.cfg,
            # we force the model to take max_len steps in each trajectory
            min_len=self.cfg.algo.max_len,
            variant=self.cfg.task.bitseq.variant,
        )

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
        "checkpoint_every": 0,
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
            "train_random_action_prob": 5e-4
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

    trial = BitSeqTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
