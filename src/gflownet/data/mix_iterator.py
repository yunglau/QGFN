import os
import sqlite3
import warnings
from collections.abc import Iterable
from copy import deepcopy
from typing import Callable, List, Tuple

import math
import networkx as nx

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from torch.utils.data import IterableDataset, Dataset
from gflownet.data.replay_buffer import ReplayBuffer

from gflownet.envs.graph_building_env import (
    GraphActionCategorical,
    GraphActionType,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
)
from gflownet.trainer import GFNAlgorithm, GFNTask

def scheduler(type, initial_prob, train_it, total_it):
    if type == 'constant':
        return initial_prob
    elif type == 'cosine_annealing':
        half_cycle = np.pi
        if train_it <= total_it:
            # Calculate the value based on cosine annealing
            step_ratio = train_it / total_it
            return (1 - np.cos(step_ratio * half_cycle)) / 2 * initial_prob
        else:
            # Return the last value of the cosine annealing phase
            return (1 - np.cos(half_cycle)) / 2 * initial_prob
    elif type == 'time_based':
        # Linear increase from initial_prob to 1 over the course of total_it iterations
        return initial_prob + (1 - initial_prob) * train_it / total_it
    elif type == 'step_based':
        step_size = 10
        increase_factor = 1.1  # Increases the probability by 10% every step
        return initial_prob * (increase_factor ** (train_it // step_size))
    elif type == 'exponential':
        growth_rate = 1.01  # Slightly greater than 1 to increase the probability
        return initial_prob * (growth_rate ** train_it)
    else:
        raise ValueError(f"Unknown scheduling type: {type}")
    
class BatchTuple:
    def __init__(self, a, b, extra_info=None):
        self.a = a
        self.b = b
        self.extra_info = extra_info

    def to(self, device):
        return BatchTuple(self.a.to(device), self.b.to(device), self.extra_info)

    def __getitem__(self, idx: int):
        if idx == 0:
            return self.a
        elif idx == 1:
            return self.b
        else:
            raise IndexError("Index must be 0 or 1")

    def __iter__(self):
        yield self.a
        yield self.b


class MixIterator(IterableDataset):
    """This iterator runs two models in sequence, and constructs batches for each model from each other's data"""

    def __init__(
        self,
        first_model: nn.Module,
        first_model_lagged: nn.Module,
        second_model: nn.Module,
        second_model_lagged: nn.Module,
        ctx: GraphBuildingEnvContext,
        first_algo: GFNAlgorithm,
        second_algo: GFNAlgorithm,
        first_task: GFNTask,
        second_task: GFNTask,
        device,
        batch_size: int,
        log_dir: str,
        stream: bool = True,
        replay_buffer: ReplayBuffer = None,
        random_action_prob: float = 0.01,
        rl_train_random_action_prob: float = 0.10, 
        p_greedy_sample: bool = False, 
        p_of_max_sample: bool = False,
        p_quantile_sample: bool = False,
        p: float = 1.0,
        scheduler_type: str = 'cosine_annealing',
        scheduler_step: int = 1500,
        ddqn_update_step: int = 1,
        hindsight_ratio: float = 0.0,
        sample_cond_info: bool = True,
        init_train_iter: int = 0,
        log_molecule_smis: bool = True,
        illegal_action_logrewards: Tuple[float, float] = (-100.0, -10.0),
    ):
        """Parameters
        ----------
        dataset: Dataset
            A dataset instance
        model: nn.Module
            The model we sample from (must be on CUDA already or share_memory() must be called so that
            parameters are synchronized between each worker)
        ctx:
            The context for the environment, e.g. a MolBuildingEnvContext instance
        algo:
            The training algorithm, e.g. a TrajectoryBalance instance
        task: GFNTask
            A Task instance, e.g. a MakeRingsTask instance
        device: torch.device
            The device the model is on
        replay_buffer: ReplayBuffer
            The replay buffer for training on past data
        batch_size: int
            The number of trajectories, each trajectory will be comprised of many graphs, so this is
            _not_ the batch size in terms of the number of graphs (that will depend on the task)
        illegal_action_logreward: float
            The logreward for invalid trajectories
        ratio: float
            The ratio of offline trajectories in the batch.
        stream: bool
            If True, data is sampled iid for every batch. Otherwise, this is a normal in-order
            dataset iterator.
        log_dir: str
            If not None, logs each SamplingIterator worker's generated molecules to that file.
        sample_cond_info: bool
            If True (default), then the dataset is a dataset of points used in offline training.
            If False, then the dataset is a dataset of preferences (e.g. used to validate the model)
        random_action_prob: float
            The probability of taking a random action, passed to the graph sampler
        init_train_iter: int
            The initial training iteration, incremented and passed to task.sample_conditional_information
        """
        self.first_model = first_model
        self.first_model_lagged = first_model_lagged
        self.second_model = second_model
        self.second_model_lagged = second_model_lagged
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        self.ctx = ctx
        self.first_algo = first_algo
        self.second_algo = second_algo
        self.first_task = first_task
        self.second_task = second_task
        self.device = device
        self.stream = stream
        self.sample_cond_info = sample_cond_info
        self.random_action_prob = random_action_prob
        self.p_greedy_sample=p_greedy_sample
        self.p_of_max_sample=p_of_max_sample
        self.p_quantile_sample=p_quantile_sample
        self.p=p
        self.rl_train_random_action_prob = rl_train_random_action_prob
        self.hindsight_ratio = hindsight_ratio
        self.train_it = init_train_iter
        self.illegal_action_logrewards = illegal_action_logrewards
        self.seed_second_trajs_with_firsts = False  # Disabled for now
        self.ddqn_update_step = ddqn_update_step
        self.scheduler_type = scheduler_type
        self.scheduler_step = scheduler_step

        # This SamplingIterator instance will be copied by torch DataLoaders for each worker, so we
        # don't want to initialize per-worker things just yet, such as where the log the worker writes
        # to. This must be done in __iter__, which is called by the DataLoader once this instance
        # has been copied into a new python process.
        self.log_dir = log_dir
        self.log = SQLiteLog()
        self.log_rl = SQLiteLog()
        self.log_hooks: List[Callable] = []

        # TODO: make this configurable depending on if the context is molecule based
        self.log_molecule_smis = False

    def add_log_hook(self, hook: Callable):
        self.log_hooks.append(hook)

    def __len__(self):
        return int(1e6)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        # Now that we know we are in a worker instance, we can initialize per-worker things
        self.rng = self.first_algo.rng = self.first_task.rng = np.random.default_rng(142857 + self._wid)
        self.ctx.device = self.device
        self.second_algo.ctx.device = self.device  # TODO: fix

        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = f"{self.log_dir}/generated_mols_{self._wid}.db"
            self.log.connect(self.log_path)

        while True:
            cond_info = self.first_task.sample_conditional_information(
                self.batch_size,
                self.train_it
            )

            p = scheduler(self.scheduler_type, self.p, self.train_it, self.scheduler_step)
            if self.p_greedy_sample: 
                p = 1 - p
                
            with torch.no_grad():                    
                first_trajs = self.first_algo.create_training_data_from_own_samples(
                    self.first_model,
                    self.second_model,
                    self.batch_size,
                    cond_info["encoding"],
                    random_action_prob=self.random_action_prob,
                    p_greedy_sample=self.p_greedy_sample,
                    p_of_max_sample=self.p_of_max_sample,
                    p_quantile_sample=self.p_quantile_sample,
                    p=p
                )
            
            all_trajs = first_trajs
            
            valid_idcs = torch.tensor(
                [i for i in range(self.batch_size) if all_trajs[i]["is_valid"]]
            ).long()

            # fetch the valid trajectories endpoints
            mols = [self.ctx.graph_to_mol(all_trajs[i]["result"]) for i in valid_idcs]

            # ask the task to compute their reward
            online_flat_rew, m_is_valid = self.first_task.compute_flat_rewards(mols)

            assert (
                online_flat_rew.ndim == 2
            ), "FlatRewards should be (mbsize, n_objectives), even if n_objectives is 1"

            # The task may decide some of the mols are invalid, we have to again filter those
            valid_idcs = valid_idcs[m_is_valid]
            pred_reward = torch.zeros((self.batch_size, online_flat_rew.shape[1]))
            pred_reward[valid_idcs] = online_flat_rew

            is_valid = torch.zeros(self.batch_size).bool()
            is_valid[valid_idcs] = True
            flat_rewards = list(pred_reward)

            # Override the is_valid key in case the task made some mols invalid
            for i in range(self.batch_size):
                all_trajs[i]["is_valid"] = is_valid[i].item()

            # Compute scalar rewards from conditional information & flat rewards
            flat_rewards = torch.stack(flat_rewards)
            first_log_rewards = self.first_task.cond_info_to_logreward(cond_info, flat_rewards)
            first_log_rewards[torch.logical_not(is_valid)] = self.illegal_action_logrewards[0]
            
            # Second task may choose to transform rewards differently
            second_log_rewards = self.second_task.cond_info_to_logreward(cond_info, flat_rewards[: self.batch_size])
            second_log_rewards[torch.logical_not(is_valid)] = self.illegal_action_logrewards[1]

            # Computes some metrics
            if self.log_dir is not None:
                if first_trajs:
                    self.log_generated(
                        deepcopy(first_trajs),
                        deepcopy(first_log_rewards),
                        deepcopy(flat_rewards),
                        {k: v for k, v in deepcopy(cond_info).items()},
                        gfn=True
                    )
            for hook in self.log_hooks:
                raise NotImplementedError()

            # if self.replay_buffer is not None: 
            #     for i in range(len(first_trajs)):
            #         self.replay_buffer.push(
            #             deepcopy(first_trajs[i])
            #         )
                    
            #     all_trajs, idxs = self.replay_buffer.sample(
            #         self.batch_size
            #     )  
                 
                
            batch_partial_traj = all_trajs
            batch_first_encoding = cond_info["encoding"]
            first_log_rewards
            if self.replay_buffer is not None: 
                # If we have a replay buffer, we push the online trajectories in it
                # and resample immediately such that the "online" data in the batch
                # comes from a more stable distribution (try to avoid forgetting)

                # cond_info is a dict, so we need to convert it to a list of dicts
                # push the online trajectories in the replay buffer and sample a new 'online' batch
                for i in range(len(first_trajs)):
                    self.replay_buffer.push(
                        deepcopy(first_trajs[i]),
                        deepcopy(first_log_rewards[i]),
                        deepcopy(flat_rewards[i]),
                        deepcopy(batch_first_encoding[i]),
                        deepcopy(is_valid[i]),
                    )
                    
                sample_output, idxs = self.replay_buffer.sample(self.batch_size)
                replay_trajs, replay_logr, replay_fr, replay_condinfo, replay_valid = sample_output

                # append the online trajectories to the offline ones
                replay_trajs = replay_trajs
                log_rewards = replay_logr
                flat_rewards = replay_fr
                cond_info = replay_condinfo
                is_valid = replay_valid

                # # convert cond_info back to a dict
                # cond_info = {k: torch.stack([d[k] for d in cond_info]) for k in cond_info[0]}
            
            # expect 64 trajectories each time for RL learning 
            batch_second_partial_traj = all_trajs
            batch_second_encoding = cond_info
            second_log_rewards 
                
            # Construct batch
            # batch = self.first_algo.construct_batch(all_trajs, torch.cat((cond_info["encoding"], cond_info_sliced["encoding"]), dim=0), first_log_rewards)
            batch = self.first_algo.construct_batch(
                replay_trajs, cond_info, log_rewards, idxs
            )
            batch.num_online = len(batch_partial_traj)
            batch.num_offline = 0
            batch.flat_rewards = flat_rewards
            second_batch = self.second_algo.construct_batch(
                            batch_second_partial_traj, batch_second_encoding, second_log_rewards
            )
            second_batch.num_online = len(batch_second_partial_traj)
            second_batch.num_offline = 0
            # self.validate_batch(self.second_model, second_batch, trajs_for_second, self.second_algo.ctx)

            self.train_it += worker_info.num_workers if worker_info is not None else 1
            bt = BatchTuple(batch, second_batch)

            bt.extra_info = {
                "first_avg_len": sum([len(i["traj"]) for i in first_trajs]) / len(first_trajs),
                "self.p": p,
            }

            yield bt

    def validate_batch(self, model, batch, trajs, ctx):
        env = GraphBuildingEnv()
        for traj in trajs:
            tp = traj["traj"] + [(traj["result"], None)]
            for t in range(len(tp) - 1):
                if tp[t][1].action == GraphActionType.Stop:
                    continue
                gp = env.step(tp[t][0], tp[t][1])
                assert nx.is_isomorphic(gp, tp[t + 1][0], lambda a, b: a == b, lambda a, b: a == b)

        for actions, atypes in [(batch.actions, ctx.action_type_order)] + (
            [(batch.bck_actions, ctx.bck_action_type_order)]
            if hasattr(batch, "bck_actions") and hasattr(ctx, "bck_action_type_order")
            else []
        ):
            mask_cat = GraphActionCategorical(
                batch,
                [model._action_type_to_mask(t, batch) for t in atypes],
                [model._action_type_to_key[t] for t in atypes],
                [None for _ in atypes],
            )
            masked_action_is_used = 1 - mask_cat.log_prob(actions, logprobs=mask_cat.logits)
            num_trajs = len(trajs)
            batch_idx = torch.arange(num_trajs, device=batch.x.device).repeat_interleave(batch.traj_lens)
            first_graph_idx = torch.zeros_like(batch.traj_lens)
            torch.cumsum(batch.traj_lens[:-1], 0, out=first_graph_idx[1:])
            if masked_action_is_used.sum() != 0:
                invalid_idx = masked_action_is_used.argmax().item()
                traj_idx = batch_idx[invalid_idx].item()
                timestep = invalid_idx - first_graph_idx[traj_idx].item()
                raise ValueError("Found an action that was masked out", trajs[traj_idx]["traj"][timestep])

    def log_generated(self, trajs, rewards, flat_rewards, cond_info, gfn=True):
        if self.log_molecule_smis:
            mols = [
                Chem.MolToSmiles(self.ctx.graph_to_mol(trajs[i]["result"])) if trajs[i]["is_valid"] else ""
                for i in range(len(trajs))
            ]
        elif hasattr(self.ctx, "object_to_log_repr"):
            mols = [self.ctx.object_to_log_repr(t["result"]) if t["is_valid"] else "" for t in trajs]
        else:
            mols = [nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(t["result"], None, "v") for t in trajs]

        flat_rewards = flat_rewards.reshape((len(flat_rewards), -1)).data.numpy().tolist()
        rewards = rewards.data.numpy().tolist()
        preferences = cond_info.get("preferences", torch.zeros((len(mols), 0))).data.numpy().tolist()
        focus_dir = cond_info.get("focus_dir", torch.zeros((len(mols), 0))).data.numpy().tolist()
        logged_keys = [k for k in sorted(cond_info.keys()) if k not in ["encoding", "preferences", "focus_dir"]]

        data = [
            [mols[i], rewards[i]]
            + flat_rewards[i]
            + preferences[i]
            + focus_dir[i]
            + [cond_info[k][i].item() for k in logged_keys]
            for i in range(len(trajs))
        ]

        data_labels = (
            ["smi", "r"]
            + [f"fr_{i}" for i in range(len(flat_rewards[0]))]
            + [f"pref_{i}" for i in range(len(preferences[0]))]
            + [f"focus_{i}" for i in range(len(focus_dir[0]))]
            + [f"ci_{k}" for k in logged_keys]
        )

        if gfn:
            self.log.insert_many(data, data_labels)
        else: 
            self.log_rl.insert_many(data, data_labels)


class SQLiteLog:
    def __init__(self, timeout=300):
        """Creates a log instance, but does not connect it to any db."""
        self.is_connected = False
        self.db = None
        self.timeout = timeout

    def connect(self, db_path: str):
        """Connects to db_path

        Parameters
        ----------
        db_path: str
            The sqlite3 database path. If it does not exist, it will be created.
        """
        self.db = sqlite3.connect(db_path, timeout=self.timeout)
        cur = self.db.cursor()
        self._has_results_table = len(
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'").fetchall()
        )
        cur.close()

    def _make_results_table(self, types, names):
        type_map = {str: "text", float: "real", int: "real"}
        col_str = ", ".join(f"{name} {type_map[t]}" for t, name in zip(types, names))
        cur = self.db.cursor()
        cur.execute(f"create table results ({col_str})")
        self._has_results_table = True
        cur.close()

    def insert_many(self, rows, column_names):
        assert all([type(x) is str or not isinstance(x, Iterable) for x in rows[0]]), "rows must only contain scalars"
        if not self._has_results_table:
            self._make_results_table([type(i) for i in rows[0]], column_names)
        cur = self.db.cursor()
        cur.executemany(f'insert into results values ({",".join("?"*len(rows[0]))})', rows)  # nosec
        cur.close()
        self.db.commit()