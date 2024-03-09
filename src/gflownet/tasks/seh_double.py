import copy
import os
import pathlib
import shutil
import socket
from typing import Any, Callable, Dict, List, Tuple, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset

from gflownet.algo.q_learning import QLearning
from gflownet.config import Config
from gflownet.data.replay_buffer import ReplayBuffer
# from gflownet.data.double_iterator import BatchTuple, DoubleIterator 
from gflownet.data.mix_iterator import BatchTuple, MixIterator
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.tasks.seh_frag import SEHTask


class SEHDoubleModelTrainer(StandardOnlineTrainer):
    task: SEHTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = True
        cfg.num_workers = 8
        cfg.checkpoint_every = 1000
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.global_batch_size = 64
        cfg.algo.offline_ratio = 0
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 9
        cfg.algo.max_edges = 128
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.01
        cfg.algo.rl_train_random_action_prob = 0.01
        cfg.algo.p_greedy_sample = False
        cfg.algo.p_of_max_sample = False
        cfg.algo.p_quantile_sample = False
        cfg.algo.p = 0.99
        cfg.algo.scheduler_type = 'cosine_annealing'
        cfg.algo.scheduler_step = 1500
        cfg.algo.ddqn_update_step = 1
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False

        cfg.replay.use = False
        cfg.replay.capacity = 5_000
        cfg.replay.warmup = 1_000
        
    def setup_algo(self):
        super().setup_algo()

        cfgp = copy.deepcopy(self.cfg)
        cfgp.algo.input_timestep = False  # Hmmm?
        cfgp.algo.illegal_action_logreward = -10
        ctxp = copy.deepcopy(self.ctx) 

        if cfgp.algo.input_timestep: 
            ctxp.num_cond_dim += 32  # Add an extra dimension for the timestep input [do we still need that?]
        if self.cfg.second_model_allow_back_and_forth:
            # Merge fwd and bck action types
            ctxp.action_type_order = ctxp.action_type_order + ctxp.bck_action_type_order
            ctxp.bck_action_type_order = ctxp.action_type_order  # Make sure the backward action types are the same
            self.second_algo.graph_sampler.compute_uniform_bck = False  # I think this might break things, to be checked
        
        self.second_algo = QLearning(self.env, ctxp, self.rng, cfgp) 
        # True is already the default, just leaving this as a reminder the we need to turn this off
        self.second_ctx = ctxp

    def setup_task(self):
        self.task = SEHTask(
            dataset=self.training_data,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )
        self.second_task = copy.copy(self.task)
        # Ignore temperature for RL task 
        self.second_task.cond_info_to_logreward = lambda cond_info, flat_reward: RewardScalar(
            flat_reward.reshape((-1,))
        )

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(num_cond_dim=self.task.num_cond_dim)

    def compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly! :)')
        
    def setup_model(self):
        super().setup_model()
        self.second_model = GraphTransformerGFN(
            self.second_ctx,
            self.cfg, 
        ) 
        
        self._get_additional_parameters = lambda: list(self.second_model.parameters())
        # # Maybe only do this if we are using DDQN?
        self.second_model_lagged = copy.deepcopy(self.second_model)
        self.second_model_lagged.to(self.device)
        self.dqn_tau = self.cfg.algo.dqn_tau
        self.ddqn_update_step = self.cfg.algo.ddqn_update_step

    def build_training_data_loader(self):
        model, dev = self._wrap_for_mp(self.sampling_model, send_to_device=True)
        gmodel, _ = self._wrap_for_mp(self.second_model, send_to_device=True)
        g_lagged_model, _ = self._wrap_for_mp(self.second_model_lagged, send_to_device=True)
        replay_buffer, _ = self._wrap_for_mp(self.replay_buffer, send_to_device=False)
        iterator = MixIterator(
            model,
            gmodel,
            g_lagged_model,
            self.ctx,
            self.algo,
            self.second_algo,
            self.task,
            self.second_task,
            dev,
            replay_buffer=replay_buffer,
            batch_size=self.cfg.algo.global_batch_size,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "train"),
            random_action_prob=self.cfg.algo.train_random_action_prob,
            p_greedy_sample=self.cfg.algo.p_greedy_sample,
            p_of_max_sample=self.cfg.algo.p_of_max_sample,
            p_quantile_sample=self.cfg.algo.p_quantile_sample,
            p=self.cfg.algo.p,
            # max_p=self.cfg.algo.max_p,
            hindsight_ratio=self.cfg.replay.hindsight_ratio,  # remove?
            illegal_action_logrewards=(
                self.cfg.algo.illegal_action_logreward,
                self.second_algo.illegal_action_logreward,
            ),
        )
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
            
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            # The 2 here is an odd quirk of torch 1.10, it is fixed and
            # replaced by None in torch 2.
            prefetch_factor=1 if self.cfg.num_workers else 2,
            # prefetch_factor=None
        )

    def train_batch(self, batch: BatchTuple, epoch_idx: int, batch_idx: int, train_it: int) -> Dict[str, Any]:
        gfn_batch, second_batch = batch
        loss, info = self.algo.compute_batch_losses(self.model, gfn_batch)
        sloss, sinfo = self.second_algo.compute_batch_losses(self.second_model, second_batch, self.second_model_lagged, temp_cond=False)
        self.step(loss + sloss, train_it)  # TODO: clip second model gradients?
        info.update({f"sec_{k}": v for k, v in sinfo.items()})
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def step(self, loss, train_it):
        super().step(loss)
        if self.dqn_tau > 0 and train_it % self.ddqn_update_step == 0:
            for a, b in zip(self.second_model.parameters(), self.second_model_lagged.parameters()):
                b.data.mul_(self.dqn_tau).add_(a.data * (1 - self.dqn_tau))

    def _save_state(self, it):
        torch.save(
            {
                "models_state_dict": [self.model.state_dict(), self.second_model.state_dict()],
                "cfg": self.cfg,
                "step": it,
            },
            open(pathlib.Path(self.cfg.log_dir) / "model_state.pt", "wb"),
        )


def main():
    """Example of how this model can be run outside of Determined"""
    hps = {
        "log_dir": "cd",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "overwrite_existing_exp": True,
        "num_training_steps": 2000,
        "validate_every": 0,
        "num_workers": 8,
        "opt": {
            "lr_decay": 20000,
        },
        "algo": {
            'p_greedy_sample': False,
            'p_of_max_sample': False,
            'p_quantile_sample': False,
            'p': 0.9,
            'dqn_tau': 0.9,
            'dqn_n_step': 3,
            'sampling_tau': 0.99,
            'global_batch_size': 64,
            'ddqn_update_step': 1,
            'rl_train_random_action_prob': 0.01,
            "tb": {"variant": "TB"},
        },
        "cond": {
            "temperature": {
                "sample_dist": "constant",
                "dist_params": [64.0],
            }
        },
        'replay': {
            'use': True,
            'capacity': 100,
            'warmup': 0,
        },
    }

    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = SEHDoubleModelTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()