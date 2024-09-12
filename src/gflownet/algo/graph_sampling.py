import copy
from typing import List, Optional
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch.nn.functional as F

from gflownet.envs.graph_building_env import Graph, GraphAction, GraphActionType
from gflownet.utils.transforms import thermometer


class GraphSampler:
    """A helper class to sample from GraphActionCategorical-producing models"""

    def __init__(
        self,
        ctx,
        env,
        max_len,
        max_nodes,
        rng,
        sample_temp=1,
        correct_idempotent=False,
        pad_with_terminal_state=False,
        input_timestep=False,
    ):
        """
        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        max_len: int
            If not None, ends trajectories of more than max_len steps.
        max_nodes: int
            If not None, ends trajectories of graphs with more than max_nodes steps (illegal action).
        rng: np.random.RandomState
            rng used to take random actions
        sample_temp: float
            Softmax temperature used when sampling, set to 0 for the greedy policy
        correct_idempotent: bool
            [Experimental] Correct for idempotent actions when counting
        pad_with_terminal_state: bool
            [Experimental] If true pads trajectories with a terminal
        """
        self.ctx = ctx
        self.env = env
        self.max_len = max_len if max_len is not None else 128
        self.max_nodes = max_nodes if max_nodes is not None else 128
        self.rng = rng
        # Experimental flags
        self.sample_temp = sample_temp
        self.sanitize_samples = True
        self.correct_idempotent = correct_idempotent
        self.pad_with_terminal_state = pad_with_terminal_state
        self.input_timestep = input_timestep
        self.compute_uniform_bck = True
        self.max_len_actual = self.max_len

    def sample_from_model(
        self,
        model: nn.Module,
        # second_model: nn.Module,
        n: int,
        cond_info: Tensor,
        dev: torch.device,
        random_action_prob: float = 0.0,
        # p_greedy_sample: bool = False, 
        # p_of_max_sample: bool = False,
        # p_quantile_sample: bool = False,
        # p: float = 0.0,
        starts: Optional[List[Graph]] = None,
    ):
        """Samples a model in a minibatch

        Parameters
        ----------
        model: nn.Module
            Model whose forward() method returns GraphActionCategorical instances
        n: int
            Number of graphs to sample
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        dev: torch.device
            Device on which data is manipulated
        starts: Optional[List[Graph]]
            If not None, a list of starting graphs. If None, starts from `self.env.new()` (typically empty graphs).

        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]], the list of states and actions
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """

        # This will be returned
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for i in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: List[List[Tensor]] = [[] for i in range(n)]
        bck_logprob: List[List[Tensor]] = [[] for i in range(n)]

        if starts is None:
            graphs = [self.env.new() for i in range(n)]
        else:
            graphs = starts
        done = [False] * n
        # TODO: instead of padding with Stop, we could have a virtual action whose probability
        # always evaluates to 1. Presently, Stop should convert to a [0,0,0] aidx, which should
        # always be at least a valid index, and will be masked out anyways -- but this isn't ideal.
        # Here we have to pad the backward actions with something, since the backward actions are
        # evaluated at s_{t+1} not s_t.
        bck_a = [[GraphAction(GraphActionType.Stop)] for i in range(n)]

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]
        
        for t in range(self.max_len):
            # Construct graphs for the trajectories that aren't yet done
            torch_graphs = [self.ctx.graph_to_Data(i, t) for i in not_done(graphs)]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            # Forward pass to get GraphActionCategorical
            # Note about `*_`, the model may be outputting its own bck_cat, but we ignore it if it does.
            # TODO: compute bck_cat.log_prob(bck_a) when relevant
            cond_info = cond_info.to(dev)
            ci = cond_info[not_done_mask]
            if self.input_timestep:
                remaining = min(1, (self.max_len - t) / self.max_len_actual)
                remaining = torch.tensor([remaining], device=dev).repeat(ci.shape[0])
                ci = torch.cat([ci, thermometer(remaining, 32)], dim=1)
                
            p_greedy_sample, p_of_max_sample, p_quantile_sample = False, False, False
            
            
            assert not (p_greedy_sample and p_of_max_sample), "Cannot sample both p_greedy_sample and p_of_max_sample"
            assert not (p_greedy_sample and p_quantile_sample), "Cannot sample both p_greedy_sample and p_quantile_sample"
            assert not (p_of_max_sample and p_quantile_sample), "Cannot sample both p_of_max_sample and p_quantile_sample"

            if p_greedy_sample: 
                logp = torch.tensor(p).log().to(dev)
                log1mp = torch.tensor(1 - p).log().to(dev)
                fwd_cat_a, *_ = model(self.ctx.collate(torch_graphs).to(dev), ci)
                fwd_cat_b, *_ = second_model(self.ctx.collate(torch_graphs).to(dev), ci)
                
                greedy_cat = copy.copy(fwd_cat_b)
                maxes = fwd_cat_b.max(fwd_cat_b.logits).values
                greedy_cat.logits = [
                    (maxes[b, None] != lg) * -1000.0 for b, lg in zip(greedy_cat.batch, greedy_cat.logits)
                ]

                mix_cat = copy.copy(fwd_cat_b)
                lp_a = fwd_cat_a.logsoftmax()
                lp_b = greedy_cat.logsoftmax()

                mix_cat.logits = [torch.logaddexp(logp + a, log1mp + b) for a, b in zip(lp_a, lp_b)]
                mix_cat.logprobs = None
                actions = mix_cat.sample()
                log_probs = mix_cat.log_prob(actions)

            elif p_quantile_sample:
                # """samples on-policy from model_a, but masks actions where model_b(s) < p * max(model_b(s))"""
                fwd_cat_a, *_ = model(self.ctx.collate(torch_graphs).to(dev), ci)
                fwd_cat_b, *_ = second_model(self.ctx.collate(torch_graphs).to(dev), ci)
                masked_cat = copy.copy(fwd_cat_b)

                # masking
                masks = []
                for logits in masked_cat.logits:
                    if logits.numel() > 0:
                        split_val = torch.quantile(logits, torch.tensor([p], dtype=logits.dtype).to(dev), dim=1)
                        expanded_split_val = split_val.unsqueeze(-1)
                        mask = logits < expanded_split_val
                        mask = mask.squeeze(0)
                        mask = mask.type(torch.bool)
                        masks.append(mask)
                    else:
                        logits = logits.type(torch.bool)
                        masks.append(logits)

                masked_cat.logits = [
                    mask * -1000.0 + ~mask * gfn_logits for mask, gfn_logits in zip(masks, fwd_cat_a.logits)
                ]

                masked_cat.logprobs = None
                actions = masked_cat.sample()
                log_probs = masked_cat.log_prob(actions)

            elif p_of_max_sample:
                fwd_cat_a, *_ = model(self.ctx.collate(torch_graphs).to(dev), ci)
                fwd_cat_b, *_ = second_model(self.ctx.collate(torch_graphs).to(dev), ci)
                
                # p_of_max_sample
                masked_cat = copy.copy(fwd_cat_b)
                maxes = torch.clamp(fwd_cat_b.max(fwd_cat_b.logits).values, min=0)
                threshold = 1e-03
                # Convert values below the threshold to zero
                maxes = torch.where(maxes < threshold, torch.zeros_like(maxes), maxes)
                masks = [(torch.clamp(Qsa, min=0) < maxes[b, None] * p) & (maxes[b, None] * p > threshold) for b, Qsa in zip(masked_cat.batch, masked_cat.logits)]
                
                masked_cat.logits = [
                    mask * -1000.0 + ~mask * gfn_logits for mask, gfn_logits in zip(masks, fwd_cat_a.logits)
                ]

                masked_cat.logprobs = None
                actions = masked_cat.sample()
                log_probs = masked_cat.log_prob(actions)

            else:
                fwd_cat, log_reward_preds = model(self.ctx.collate(torch_graphs).to(dev), ci)
                
                # print(test)
                # quit()
                if random_action_prob > 0:
                    masks = [1] * len(fwd_cat.logits) if fwd_cat.masks is None else fwd_cat.masks
                    # Device which graphs in the minibatch will get their action randomized
                    is_random_action = torch.tensor(
                        self.rng.uniform(size=len(torch_graphs)) < random_action_prob, device=dev
                    ).float()
                    # Set the logits to some large value if they're not masked, this way the masked
                    # actions have no probability of getting sampled, and there is a uniform
                    # distribution over the rest
                    fwd_cat.logits = [
                        # We don't multiply m by i on the right because we're assume the model forward()
                        # method already does that
                        is_random_action[b][:, None] * torch.ones_like(i) * m * 100 + i * (1 - is_random_action[b][:, None])
                        for i, m, b in zip(fwd_cat.logits, masks, fwd_cat.batch)
                    ]
                    
                if self.sample_temp != 1:
                    sample_cat = copy.copy(fwd_cat)
                    sample_cat.logits = [i / self.sample_temp for i in fwd_cat.logits]
                    actions = sample_cat.sample()
                else:
                    actions = fwd_cat.sample()
                    
                log_probs = fwd_cat.log_prob(actions)
            
            # actions = action_callback(self.ctx.collate(torch_graphs).to(dev), ci)
            graph_actions = [self.ctx.aidx_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions)]
            # Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n)):
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]["traj"].append((graphs[i], graph_actions[j]))
                if self.compute_uniform_bck:
                    bck_a[i].append(self.env.reverse(graphs[i], graph_actions[j]))
                # Check if we're done
                if graph_actions[j].action is GraphActionType.Stop:
                    done[i] = True
                    if self.compute_uniform_bck:
                        bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                    data[i]["is_sink"].append(1)
                else:  # If not done, try to step the self.environment
                    gp = graphs[i]
                    try:
                        # self.env.step can raise AssertionError if the action is illegal
                        gp = self.env.step(graphs[i], graph_actions[j])
                        assert len(gp.nodes) <= self.max_nodes
                    except AssertionError:
                        done[i] = True
                        data[i]["is_valid"] = False
                        if self.compute_uniform_bck:
                            bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                        data[i]["is_sink"].append(1)
                        continue
                    if t == self.max_len - 1:
                        done[i] = True
                    # If no error, add to the trajectory
                    if self.compute_uniform_bck:
                        # P_B = uniform backward
                        n_back = self.env.count_backward_transitions(gp, check_idempotent=self.correct_idempotent)
                        bck_logprob[i].append(torch.tensor([1 / n_back], device=dev).log())
                    data[i]["is_sink"].append(0)
                    graphs[i] = gp
                if done[i] and self.sanitize_samples and not self.ctx.is_sane(graphs[i]):
                    # check if the graph is sane (e.g. RDKit can
                    # construct a molecule from it) otherwise
                    # treat the done action as illegal
                    data[i]["is_valid"] = False
            if all(done):
                break

        # is_sink indicates to a GFN algorithm that P_B(s) must be 1

        # There are 3 types of possible trajectories
        #  A - ends with a stop action. traj = [..., (g, a), (gp, Stop)], P_B = [..., bck(gp), 1]
        #  B - ends with an invalid action.  = [..., (g, a)],                 = [..., 1]
        #  C - ends at max_len.              = [..., (g, a)],                 = [..., bck(gp)]

        # Let's say we pad terminal states, then:
        #  A - ends with a stop action. traj = [..., (g, a), (gp, Stop), (gp, None)], P_B = [..., bck(gp), 1, 1]
        #  B - ends with an invalid action.  = [..., (g, a), (g, None)],                  = [..., 1, 1]
        #  C - ends at max_len.              = [..., (g, a), (gp, None)],                 = [..., bck(gp), 1]
        # and then P_F(terminal) "must" be 1
        for i in range(n):
            # If we're not bootstrapping, we could query the reward
            # model here, but this is expensive/impractical.  Instead
            # just report forward and backward logprobs
            data[i]["fwd_logprob"] = sum(fwd_logprob[i])
            data[i]["result"] = graphs[i]
            if self.compute_uniform_bck:
                data[i]["bck_logprob"] = sum(bck_logprob[i])
                data[i]["bck_logprobs"] = torch.stack(bck_logprob[i]).reshape(-1)
                data[i]["bck_a"] = bck_a[i]
            if self.pad_with_terminal_state:
                # TODO: instead of padding with Stop, we could have a virtual action whose
                # probability always evaluates to 1.
                data[i]["traj"].append((graphs[i], GraphAction(GraphActionType.Stop)))
                data[i]["is_sink"].append(1)
        return data