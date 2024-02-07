"""
Generic online trainer for mixture policies that require multiple models
"""

import copy
import pathlib

import torch
from typing import Any, Dict

from gflownet.data.mix_iterator import BatchTuple, MixIterator
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.models.seq_transformer import SeqTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer

from gflownet.algo.q_learning import QLearning


class MixtureOnlineTrainer(StandardOnlineTrainer):

    def setup_algo(self):
        super().setup_algo()

        cfgp = copy.deepcopy(self.cfg)
        cfgp.algo.input_timestep = False  # Hmmm?
        cfgp.algo.illegal_action_logreward = -10
        ctxp = copy.deepcopy(self.ctx)

        if cfgp.algo.input_timestep:
            # Add an extra dimension for the timestep input [do we still need that?] 
            ctxp.num_cond_dim += 32
        if self.cfg.second_model_allow_back_and_forth:
            # Merge fwd and bck action types
            ctxp.action_type_order = ctxp.action_type_order + ctxp.bck_action_type_order
            ctxp.bck_action_type_order = ctxp.action_type_order  # Make sure the backward action types are the same
            self.second_algo.graph_sampler.compute_uniform_bck = False  # I think this might break things, to be checked
        
        self.second_algo = QLearning(self.env, ctxp, self.rng, cfgp) 
        # True is already the default, just leaving this as a reminder the we need to turn this off
        self.second_ctx = ctxp

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
        # Maybe only do this if we are using DDQN?
        self.second_model_lagged = copy.deepcopy(self.second_model)
        self.second_model_lagged.to(self.device)
        self.dqn_tau = self.cfg.algo.dqn_tau
        self.ddqn_update_step = self.cfg.algo.ddqn_update_step

    def build_training_data_loader(self):
        model, dev = self._wrap_for_mp(self.sampling_model, send_to_device=True)
        gmodel, dev = self._wrap_for_mp(self.second_model, send_to_device=True)
        g_lagged_model, dev = self._wrap_for_mp(self.second_model_lagged, send_to_device=True)
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
            scheduler_type=self.cfg.algo.scheduler_type,
            scheduler_step=self.cfg.algo.scheduler_step,
            hindsight_ratio=self.cfg.replay.hindsight_ratio,  # remove?
            illegal_action_logrewards=(
                self.cfg.algo.illegal_action_logreward,
                self.second_algo.illegal_action_logreward,
            )
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
