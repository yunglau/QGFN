import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor
from torch_scatter import scatter

from gflownet.algo.graph_sampling import GraphSampler
from gflownet.config import Config
from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphBuildingEnvContext, generate_forward_trajectory


class SoftQLearning:
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
        self.gamma = cfg.algo.sql.gamma
        self.invalid_penalty = cfg.algo.sql.penalty
        self.bootstrap_own_reward = False
        # Munchausen-DQN params
        self.is_munchausen = True
        self.entropy_coeff = 1 / (1 - self.alpha)
        self.m_l0 = -2500.0
        # Experimental flags
        self.sample_temp = 1
        self.do_q_prime_correction = False
        self.graph_sampler = GraphSampler(ctx, env, self.max_len, self.max_nodes, rng, self.sample_temp)

    def create_training_data_from_own_samples(
        self, model: nn.Module, second_model: nn.Module, batch_size: int, cond_info: Tensor, random_action_prob: float = 0.0, p_greedy_sample: bool = False, p_of_max_sample: bool = False, p_quantile_sample: bool = False, scale_temp: bool=False, p: float = 1.0,
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
        data = self.graph_sampler.sample_from_model(model, second_model, batch_size, cond_info, dev, random_action_prob, p_greedy_sample, p_of_max_sample, p_quantile_sample, p)
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
        trajs = [{"traj": generate_forward_trajectory(i)} for i in graphs]
        for traj in trajs:
            n_back = [
                self.env.count_backward_transitions(gp, check_idempotent=self.cfg.do_correct_idempotent)
                for gp, _ in traj["traj"][1:]
            ] + [1]
            traj["bck_logprobs"] = (1 / torch.tensor(n_back).float()).log().to(self.ctx.device)
            traj["result"] = traj["traj"][-1][0]
        return trajs

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
            self.ctx.GraphAction_to_aidx(g, a)
            for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj["traj"]])
        ]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.log_p_B = torch.cat([i["bck_logprobs"] for i in trajs], 0)
        batch.actions = torch.tensor(actions)
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()
        return batch

    def compute_batch_losses(
        self,
        model: nn.Module,
        target_model: nn.Module,
        batch: gd.Batch,
        num_bootstrap: int = 0
    ):
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
        num_trajs = int(batch.traj_lens.shape[0])
        rewards = torch.exp(batch.log_rewards)
        cond_info = batch.cond_info

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        # The position of the last graph of each trajectory
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # Forward pass of the model, returns a GraphActionCategorical and per molecule predictions
        # Here we will interpret the logits of the fwd_cat as Q values
        Q, per_state_preds = model(batch, cond_info[batch_idx])
        policy, _ = target_model(batch, cond_info[batch_idx])

        if self.do_q_prime_correction:
            # First we need to estimate V_soft. We will use q_a' = \pi
            log_policy = Q.logsoftmax()
            # in Eq (10) we have an expectation E_{a~q_a'}[exp(1/alpha Q(s,a))/q_a'(a)]
            # we rewrite the inner part `exp(a)/b` as `exp(a-log(b))` since we have the log_policy probabilities
            soft_expectation = [Q_sa / self.alpha - logprob for Q_sa, logprob in zip(Q.logits, log_policy)]
            # This allows us to more neatly just call logsumexp on the logits, and then multiply by alpha
            V_soft = self.alpha * Q.logsumexp(soft_expectation).detach()  # shape: (num_graphs,)
        elif self.is_munchausen: 
            target_log_policy = policy.log_prob(batch.actions, logprobs=policy.logits).detach()
            munchausen_penalty = torch.clamp(
                self.entropy_coeff * target_log_policy,
                min=self.m_l0, max=1
            )
            V_soft = Q.logsumexp(Q.logits).detach() + self.alpha * munchausen_penalty
            V_soft = V_soft + batch.log_p_B
            # MDQN loss uses log rewards, we clip these
            log_rewards = batch.log_rewards
            assert log_rewards.ndim == 1
            clip_log_R = torch.maximum(log_rewards, torch.tensor(
                self.illegal_action_logreward, device=dev)).float()
            rewards = clip_log_R
        else:
            V_soft = Q.logsumexp(Q.logits).detach() 
            rewards = rewards / self.alpha

        # Here were are again hijacking the GraphActionCategorical machinery to get Q[s,a], but
        # instead of logprobs we're just going to use the logits, i.e. the Q values.
        Q_sa = Q.log_prob(batch.actions, logprobs=Q.logits)

        # We now need to compute the target, \hat Q = R_t + V_soft(s_t+1)
        # Shift t+1-> t, pad last state with a 0, multiply by gamma
        shifted_V_soft = self.gamma * torch.cat([V_soft[1:], torch.zeros_like(V_soft[:1])])
        # Replace V(s_T) with R(tau). Since we've shifted the values in the array, V(s_T) is V(s_0)
        # of the next trajectory in the array, and rewards are terminal (0 except at s_T).
        shifted_V_soft[final_graph_idx] = rewards + (1 - batch.is_valid) * self.invalid_penalty
        # The result is \hat Q = R_t + gamma V(s_t+1)
        hat_Q = shifted_V_soft

        losses = (Q_sa - hat_Q).pow(2)
        traj_losses = scatter(losses, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        loss = losses.mean()
        invalid_mask = 1 - batch.is_valid
        info = {
            "mean_loss": loss,
            "offline_loss": traj_losses[: batch.num_offline].mean() if batch.num_offline > 0 else 0,
            "online_loss": traj_losses[batch.num_offline :].mean() if batch.num_online > 0 else 0,
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            "invalid_losses": (invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4),
        }

        if not torch.isfinite(traj_losses).all():
            raise ValueError("loss is not finite")
        return loss, info
