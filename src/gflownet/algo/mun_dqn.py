import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor
from torch_scatter import scatter
import torch.nn.functional as F

from gflownet.algo.graph_sampling import GraphSampler
from gflownet.config import Config
from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphBuildingEnvContext, generate_forward_trajectory

import pdb

class Munchausen_DQN:
    def __init__(
        self,
        env: GraphBuildingEnv,
        ctx: GraphBuildingEnvContext,
        rng: np.random.RandomState,
        cfg: Config,
    ):
        """Soft Q-Learning implementation, see
        Haarnoja, Tuomas, Haoran Tang, Pieter Abbeel, and Sergey Levine. "Reinforcement learning with deep
        energy-based policies." In International conference on machine learning, pp. 1352-1361. PMLR, 2017.

        Hyperparameters used:
        illegal_action_logreward: float, log(R) given to the model for non-sane end states or illegal actions

        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        rng: np.random.RandomState
            rng used to take random actions
        cfg: Config
            The experiment configuration
        """
        self.ctx = ctx
        self.env = env
        self.rng = rng
        self.max_len = cfg.algo.max_len
        self.max_nodes = cfg.algo.max_nodes
        self.illegal_action_logreward = cfg.algo.illegal_action_logreward
        self.alpha = cfg.algo.sql.alpha
        self.entropy_coefficient = (1/(1-self.alpha))
        self.gamma = cfg.algo.sql.gamma
        self.invalid_penalty = cfg.algo.sql.penalty
        self.bootstrap_own_reward = False
        # Experimental flags
        self.sample_temp = 1
        self.do_q_prime_correction = False
        self.graph_sampler = GraphSampler(ctx, env, self.max_len, self.max_nodes, rng, self.sample_temp)

    def create_training_data_from_own_samples(
        self, model: nn.Module, n: int, cond_info: Tensor, random_action_prob: float
    ):
        """Generate trajectories by sampling a model

        Parameters
        ----------
        model: nn.Module
           The model being sampled
        graphs: List[Graph]
            List of N Graph endpoints
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
        random_action_prob: float
            Probability of taking a random action
        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]]
           - fwd_logprob: log Z + sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        dev = self.ctx.device
        cond_info = cond_info.to(dev)
        data = self.graph_sampler.sample_from_model(model, n, cond_info, dev, random_action_prob)
        return data

    def create_training_data_from_graphs(self, graphs):
        """Generate trajectories from known endpoints

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints

        Returns
        -------
        trajs: List[Dict{'traj': List[tuple[Graph, GraphAction]]}]
           A list of trajectories.
        """
        return [{"traj": generate_forward_trajectory(i)} for i in graphs]

    def construct_batch(self, trajs, cond_info, log_rewards):
        """Construct a batch from a list of trajectories and their information

        Parameters
        ----------
        trajs: List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories.
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        log_rewards: Tensor
            The transformed log-reward (e.g. torch.log(R(x) ** beta) ) for each trajectory. Shape (N,)
        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """
        torch_graphs = [self.ctx.graph_to_Data(i[0]) for tj in trajs for i in tj["traj"]]
        actions = [
            self.ctx.GraphAction_to_aidx(g, a) for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj["traj"]])
        ]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.actions = torch.tensor(actions)
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()
        return batch

    def construct_batch_transitions(self, transitions, cond_info, idxs):
        """Construct a batch from a list of trajectories and their information

        Parameters
        ----------
        trajs: list of tuples 
            A list of tuple containing states, actions, next_states, log_rewards, and done
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """
        states, actions, log_rewards, next_states, dones, valid_traj = transitions
        torch_graphs = [self.ctx.graph_to_Data(i) for i in states]
        torch_graphs_next_states = [self.ctx.graph_to_Data(i) for i in next_states]
        actions = [
            self.ctx.GraphAction_to_aidx(g, a) for g, a in zip(torch_graphs, actions)
        ]
        
        # bck_actions = [
        #     self.ctx.GraphAction_to_aidx(g, a) for g, a in zip(torch_graphs, backward_actions)
        # ]
        
        batch = self.ctx.collate(torch_graphs)
        batch.next_states = self.ctx.collate(torch_graphs_next_states)
        batch.actions = torch.tensor(actions)
        batch.log_rewards = log_rewards
        batch.dones = dones
        batch.cond_info = cond_info
        # batch.bck_actions = bck_actions
        batch.is_valid = valid_traj
        # batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()
        batch.transitions = transitions
        batch.idxs = idxs
        return batch

    def compute_batch_losses(self, model: nn.Module, target_model: nn.Module, batch: gd.Batch, num_bootstrap: int = 0):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: TrajectoryBalanceModel
           A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
           Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch: gd.Batch
          batch of graphs inputs as per constructed by `self.construct_batch`
        num_bootstrap: int
          the number of trajectories for which the reward loss is computed. Ignored if 0."""
        dev = batch.x.device
        # A single trajectory is comprised of many graphs
        # num_trajs = int(batch.traj_lens.shape[0])
        # rewards = torch.exp(batch.log_rewards)
        mdp_rewards = batch.log_rewards
        cond_info = batch.cond_info
        dones = batch.dones
        dones = torch.tensor(dones).cuda()

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        # batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        # The position of the last graph of each trajectory
        # final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # Forward pass of the model, returns a GraphActionCategorical and per molecule predictions
        # Here we will interpret the logits of the fwd_cat as Q values
        is_valid = batch.is_valid.float().clone().detach().to('cuda')
        Q, per_state_preds = model(batch, cond_info)
        with torch.no_grad():
            Q_target_st, _ = target_model(batch, cond_info)
            Qp_target_next_st, _ = target_model(batch.next_states, cond_info)
        
        # Here were are again hijacking the GraphActionCategorical machinery to get Q[s,a], but
        # instead of logprobs we're just going to use the logits, i.e. the Q values.
        Q_sa = Q.log_prob(batch.actions, logprobs=Q.logits)
        
        # log_p_B = bck_cat.log_prob(batch.bck_actions)
        
        second_term = (1 - dones) * Qp_target_next_st.logsumexp(Qp_target_next_st.logits).detach()
        
        # Now we are computing the third term 
        # Compute the log probabilities of the actions using the policy logits
        munchausen_term = self.entropy_coefficient * Q_target_st.log_prob(batch.actions).detach()
        
        # print(munchausen_term)
        
        # Clamp the Munchausen term to a specified range
        munchausen_penalty = self.alpha * torch.clamp(munchausen_term, min=torch.tensor(-2500).to('cuda'), max=torch.tensor(1).to('cuda'))
        
        # pdb.set_trace()
        
        # print("munchausen_penalty")
        # print(munchausen_penalty)
        
        hat_Q = mdp_rewards.clone().detach().to('cuda') + second_term + munchausen_term
        
        losses = nn.functional.huber_loss(Q_sa, hat_Q, reduction="none") # used to update priority 
        
        loss = losses.mean()
        # print(loss)
        invalid_mask = 1 - is_valid
        info = {
            "mean_loss": loss,
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            # "invalid_losses": (invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4),
            "Q_sa": Q_sa.mean().item(),
        }

        return loss, info, losses