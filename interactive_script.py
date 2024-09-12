import torch
from gflownet.tasks.seh_double import SEHDoubleModelTrainer
# # from gflownet.tasks.qm9.qm9_double import QM9MixtureModelTrainer
# from gflownet.tasks.bitseq.bitseq_mix import BitSeqMixTrainer

log_root = '/network/scratch/e/elaine.lau/mun_dqn/sampling_tau_15_beta32_replay_50000'

base_hps = {
    'log_dir': log_root,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    'overwrite_existing_exp': True,
    'num_training_steps': 10000,
    'validate_every': 0,
    'num_workers': 0,
    'opt': {
        'lr_decay': 2000
    },
    'algo': {
        'method': 'MUNDQN',
        'p_greedy_sample': False,
        'p_of_max_sample': False,
        'p_quantile_sample': False,
        'scheduler_step': 1500,
        'scheduler_type': 'cosine_annealing',
        'p': 0.4,
        'dqn_n_step': 30,
        'sampling_tau': 0.9,
        'global_batch_size': 64,
        'ddqn_update_step': 1,
        'train_random_action_prob': 0.05,
        'rl_train_random_action_prob': 0.05,
        'dqn_tau': 0.9
    },
    'cond': {
        'temperature': {
            'sample_dist': 'constant',
            'dist_params': [32.0],
            'num_thermometer_dim': 32,
        }
    },
    'replay': {
        'use': True,
        'capacity': 1_000_000,
        'warmup': 4096,
    },
    "task": {
        "qm9": {
            "h5_path": "path.to.dataset/qm9.h5",
            "model_path": "path.to.model/mxmnet_gap_model.pt"
        },
        "bitseq": {
            "variant": "prepend-append", # "autoregressive" or "prepend-append"
            "modes_path": "data/modes.pkl",
            "k": 4,
        }
    }
}

trial = SEHDoubleModelTrainer(base_hps)
trial.print_every = 1
trial.run()

# import pdb
# pdb.set_trace()