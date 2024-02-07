import torch
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import os
from itertools import count
from matplotlib import cm
import matplotlib.pyplot as pp
import pandas as pd
from gflownet.tasks.seh_double import SEHDoubleModelTrainer
from gflownet.envs.graph_building_env import *
import tqdm
from rdkit import Chem, DataStructs
import json
import glob

def compute_pairwise_sims(mols):
    fps = [Chem.RDKFingerprint(x) for x in mols]
    sims = []
    for j in range(len(fps)-1):
        sims += DataStructs.BulkTanimotoSimilarity(fps[j], fps[j+1:])
    return sims

def is_new_mode(obj, modes, sim_threshold=0.7):
    """
    Returns True if obj is a new mode, False otherwise.
    """      
    # mol_obj = Chem.MolFromSmiles(obj)
    obj_id = Chem.RDKFingerprint(obj)

    if len(modes) == 0:
        return True, obj_id

    if obj_id is None:
        return False, None
    
    sim_scores = DataStructs.BulkTanimotoSimilarity(obj_id, modes)
    return all(s < sim_threshold for s in sim_scores), obj_id

    
def get_modes(rewards, smiles, min_reward: float=0.9, sim_threshold: float=0.7, save_df: bool=False) -> pd.DataFrame:
    """
    Returns number of modes given rewards and smiles 
    """
    modes = []      # Store unique mode ids
    num_modes = []  # Store the number of unique modes at each step
    
    for j in range(1, len(smiles)):

        if rewards[j] < min_reward or smiles[j] is None:
            num_modes.append(len(modes))
            continue

        new_mode, obj_id = is_new_mode(\
            smiles[j], modes, sim_threshold=sim_threshold)

        if new_mode:
            modes.append(obj_id)
        
        num_modes.append(len(modes))

    return num_modes[-1]

trainer = SEHDoubleModelTrainer({
    "log_dir": "./log_tmp/",
    "device": "cuda",
    "num_training_steps": 2000,
    "validate_every": 0,
    "num_workers": 0,
    "algo": {
        "sampling_tau": 0.95,
        "global_batch_size": 64,
        "tb": {"variant": "SubTB1"},
    },
    "cond": {
        "temperature": {
            "sample_dist": "constant",
            "dist_params": [32.0],
        }
    },
})

# root_dir = f"/rxrx/data/user/elaine.lau/recursion/TB_baseline_batch64_beta32/seed-{seed}/"
root_dir = f"/rxrx/data/user/elaine.lau/recursion/jobs/effect_of_alpha2023-12-27/63f1fa06-30ae-40e0-b243-5dd1cf8493f0-20-15-55/seed-1/"
state = torch.load(f'{root_dir}/model_state.pt')

trainer.model.load_state_dict(state['models_state_dict'][0])
trainer.model.cpu()
trainer.second_model.load_state_dict(state['models_state_dict'][1])
trainer.second_model.cpu()

trainer.algo.ctx.device = torch.device('cpu')


def run_experiment(seed):
    trainer = SEHDoubleModelTrainer({
        "log_dir": "./log_tmp/",
        "device": "cpu",
        "num_training_steps": 2000,
        "validate_every": 0,
        "num_workers": 0,
        "algo": {
            "sampling_tau": 0.95,
            "global_batch_size": 64,
            "tb": {"variant": "SubTB1"},
        },
        "cond": {
            "temperature": {
                "sample_dist": "constant",
                "dist_params": [32.0],
            }
        },
    })
    
    n_samples = 256
    cond_info = {}
    cond_info['beta'] = torch.zeros((n_samples, 32)).cpu()
    cond_info['encoding'] = torch.zeros((n_samples, 32)).cpu()

    trajs = trainer.algo.create_training_data_from_own_samples(trainer.model, trainer.model, n_samples, cond_info, 0.01)
    rewards = trainer.task.compute_flat_rewards([trainer.ctx.graph_to_mol(i['result']) for i in trajs])[0].flatten()
    mols = [trainer.ctx.graph_to_mol(i['result']) for i in trajs]
    sim_dist = np.float32(compute_pairwise_sims(mols))

    return run_experiment, sim_dist


def high_Q_sample(model_a, model_b, p):
    """samples on-policy from model_a, but masks actions where model_b(s) < p * max(model_b(s))"""
    logp = torch.tensor(p).log().cpu()
    log1mp = torch.tensor(1 - p).log().cpu()
    def callback(batch, ci):
        fwd_cat_a, *_ = model_a(batch, ci)
        fwd_cat_b, *_ = model_b(batch, ci)

        masked_cat = copy.copy(fwd_cat_b)
        maxes = fwd_cat_b.max(fwd_cat_b.logits).values
        masks = [(Qsa < maxes[b, None] * p) for b, Qsa in zip(masked_cat.batch, masked_cat.logits)]
        
        masked_cat.logits = [
             mask * -1000.0 + ~mask * gfn_logits for mask, gfn_logits in zip(masks, fwd_cat_a.logits)
        ]
        #print(maxes)
        #print(fwd_cat_a.logits)
        #print(fwd_cat_b.logits)
        #print(masked_cat.logits)
        #print(masks, fwd_cat_a.masks)
        #print(sum([((i == 1) != (j == 0)).sum() for i,j in zip(masks, fwd_cat_a.masks)]))
        masked_cat.logprobs = None
        return masked_cat.sample()
    return callback

def top_p_sample(model_a, model_b, p):
    """samples on-policy from model_a, but masks actions where model_b(s) < p * max(model_b(s))"""
    logp = torch.tensor(p).log().cpu()
    log1mp = torch.tensor(1 - p).log().cpu()
    def callback(batch, ci):
        fwd_cat_a, *_ = model_a(batch, ci)
        fwd_cat_b, *_ = model_b(batch, ci)

        masked_cat = copy.copy(fwd_cat_b)
        masks = []
        for logits in masked_cat.logits:
            if logits.numel() > 0:
                split_val = torch.quantile(logits, torch.tensor([p], dtype=logits.dtype).to('cpu'), dim=1)
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
        #print(maxes)
        #print(fwd_cat_a.logits)
        #print(fwd_cat_b.logits)
        #print(masked_cat.logits)
        #print(masks, fwd_cat_a.masks)
        #print(sum([((i == 1) != (j == 0)).sum() for i,j in zip(masks, fwd_cat_a.masks)]))
        masked_cat.logprobs = None
        return masked_cat.sample()
    return callback

def logit_sampler(model):
    def callback(batch, ci):
        fwd_cat, *_ = model(batch, ci)
        return fwd_cat.sample()
    return callback


def p_greedy_sample(model_a, model_b, p):
    """samples on-policy from model_a, greedily from model_b, with probability p, 1-p respectively"""
    logp = torch.tensor(p).log().to('cpu')
    log1mp = torch.tensor(1 - p).log().to('cpu')
    def callback(batch, ci):
        fwd_cat_a, *_ = model_a(batch, ci)
        fwd_cat_b, *_ = model_b(batch, ci)

        greedy_cat = copy.copy(fwd_cat_b)
        maxes = fwd_cat_b.max(fwd_cat_b.logits).values
        greedy_cat.logits = [
            (maxes[b, None] != lg) * -1000.0 for b, lg in zip(greedy_cat.batch, greedy_cat.logits)
        ]
        #print(greedy_cat.logits)
        mix_cat = copy.copy(fwd_cat_b)
        lp_a = fwd_cat_a.logsoftmax()
        lp_b = greedy_cat.logsoftmax()
        #print(lp_a)
        #print(lp_b)
        mix_cat.logits = [torch.logaddexp(logp + a, log1mp + b) for a, b in zip(lp_a, lp_b)]
        mix_cat.logprobs = None
        #print(mix_cat.logsoftmax())
        return mix_cat.sample()
    return callback
        
def epsilon_greedy_sampler(model, random_action_prob):
    def callback(batch, ci):
        fwd_cat, *_ = model(batch, ci)
        if random_action_prob > 0:
            masks = [1] * len(fwd_cat.logits) if fwd_cat.masks is None else fwd_cat.masks
            # Device which graphs in the minibatch will get their action randomized
            is_random_action = torch.tensor(
                rng.uniform(size=batch.num_graphs) < random_action_prob, device='cpu'
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

        sample_cat = copy.copy(fwd_cat)
        maxes = fwd_cat.max(fwd_cat.logits).values
        sample_cat.logits = [
            (maxes[b, None] != lg) * -1000.0 for b, lg in zip(fwd_cat.batch, fwd_cat.logits)
        ]
        return sample_cat.sample()
    return callback

def generic_graph_sampling(action_callback, n):
    #### 
    self = trainer.algo.graph_sampler
    starts = None
    cond_info = torch.zeros((n, 32)).to('cpu')
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
        not_done_mask = torch.tensor(done, device='cpu').logical_not()
        # Forward pass to get GraphActionCategorical
        # Note about `*_`, the model may be outputting its own bck_cat, but we ignore it if it does.
        # TODO: compute bck_cat.log_prob(bck_a) when relevant
        ci = cond_info[not_done_mask]
        actions = action_callback(self.ctx.collate(torch_graphs).to('cpu'), ci)
        graph_actions = [self.ctx.aidx_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions)]
        # Step each trajectory, and accumulate statistics
        for i, j in zip(not_done(range(n)), range(n)):
            data[i]["traj"].append((graphs[i], graph_actions[j]))
            if graph_actions[j].action is GraphActionType.Stop:
                done[i] = True
            else:
                gp = graphs[i]
                try:
                    gp = self.env.step(graphs[i], graph_actions[j])
                    assert len(gp.nodes) <= self.max_nodes
                except AssertionError:
                    done[i] = True
                    continue
                if t == self.max_len - 1:
                    done[i] = True
                graphs[i] = gp
        if all(done):
            break

    for i in range(n):
        data[i]["result"] = graphs[i]
    return data

def process_directory(root_dir, p, function):
    high_Q_reward_dist = []
    high_Q_mol_dist = []
    all_modes = []
    n_samples = 1000
    try:
        print(root_dir)
        for seed in range(1, 6):
            root_seed_dir = f"{root_dir}/seed-{seed}/"
            state = torch.load(f'{root_seed_dir}/model_state.pt')

            trainer.model.load_state_dict(state['models_state_dict'][0])
            trainer.model.cpu()
            trainer.second_model.load_state_dict(state['models_state_dict'][1])
            trainer.second_model.cpu()
            
            with torch.no_grad():
                trajs = generic_graph_sampling(function(trainer.model, trainer.second_model, p), n_samples)
                mols = [trainer.ctx.graph_to_mol(i['result']) for i in trajs]
                rewards = trainer.task.compute_flat_rewards(mols)[0].flatten()
            
            modes = get_modes(rewards, mols)
            
            all_modes.append(modes)
            high_Q_reward_dist.append(rewards)
            high_Q_sim_dist = np.float32(compute_pairwise_sims(mols))
            high_Q_mol_dist.append(high_Q_sim_dist)
    except: 
        print(root_seed_dir)

    all_means = np.array([r.mean().item() for r in high_Q_reward_dist])
    rewards_mean = all_means.mean()
    reward_std_err = all_means.std(ddof=1) / np.sqrt(len(all_means))

    all_sim_means = np.array([r.mean().item() for r in high_Q_mol_dist])
    similarity_mean = all_sim_means.mean()
    similarity_std_err = all_sim_means.std(ddof=1) / np.sqrt(len(all_sim_means))
    
    return rewards_mean, similarity_mean, reward_std_err, similarity_std_err, np.mean(all_modes)


# Main processing loop
def main():
    root_path = '/rxrx/data/user/elaine.lau/recursion/jobs/sanity_check'
    all_results = []
    file_names = ['./high_Q_sample_sanity_check.csv']
    
    # Iterate through each subdirectory
    for csv_name in file_names: 
        all_results = []
        for subdir in glob.glob(os.path.join(root_path, '*/')):
            # Load the run_object.json file
                with open(os.path.join(subdir, 'run_object.json'), 'r') as file:
                    run_object = json.load(file)
                    beta_value = run_object['hparams']['temperature']  # Replace 'beta' with the correct key if different
                    dqn_n_step = run_object['hparams']['dqn_n_step']
                    p = run_object['hparams']['p']

                if 'baseline' in csv_name: 
                    # Process the directory
                    p = 0.0
                    avg_reward, avg_similarity, reward_std_err, similarity_std_err, modes = process_directory(subdir, p, top_p_sample)
                elif 'top_p_sample' in csv_name: 
                    all_p = [p]
                    # all_p = np.linspace(0.8, 0.95, 20)
                    for p in all_p: 
                        
                        # Process the directory
                        avg_reward, avg_similarity, reward_std_err, similarity_std_err, modes = process_directory(subdir, p, top_p_sample)
                elif 'high_Q_sample' in csv_name: 
                    # all_p = np.linspace(0.85, 0.99, 20)
                    all_p = [p]
                    for p in all_p:
                        # Process the directory
                        avg_reward, avg_similarity, reward_std_err, similarity_std_err, modes = process_directory(subdir, p, high_Q_sample)
                elif 'p_greedy_sample' in csv_name: 
                    # all_p = np.linspace(0.4, 0.75, 20)
                    all_p = [p]
                    for p in all_p:
                        # Process the directory
                        avg_reward, avg_similarity, reward_std_err, similarity_std_err, modes = process_directory(subdir, p, p_greedy_sample)
                
                all_results.append([beta_value, dqn_n_step, p, avg_reward, avg_similarity, reward_std_err, similarity_std_err, modes, subdir])
                
        # Convert to DataFrame
        df = pd.DataFrame(all_results, columns=['Beta Value', 'dqn_n_step', 'p_value', 'Rewards', 'Similarity', 'reward_std_err', 'similarity_std_err', 'modes', 'file_path'])

        # Save DataFrame to CSV
        df.to_csv(csv_name, index=False, mode='a')

if __name__ == "__main__":
    main()
