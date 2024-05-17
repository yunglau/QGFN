"""
Class methods for generating runs
"""

import os
import json
from uuid import uuid4
from typing import TypedDict
from datetime import datetime
from gflownet.config import Config


class HParams(TypedDict):
    temperature: float
    use_replay: bool
    tb_variant: str
    ddqn_update_step: int
    buffer_size: int
    dqn_tau: float
    dqn_epsilon: float
    batch_size: int
    num_workers: int
    num_training_steps: int


make_py_script = lambda task, hps, log_dir, seeds=5: f"""
import sys, os
from gflownet.tasks.seh_double import SEHDoubleModelTrainer

base_hps = {hps}

config = [
    {{
        **base_hps,
        'log_dir': f'{log_dir}/seed-{{seed+1}}/',
    }}
    for seed in range({seeds})
]

if __name__ == '__main__':
    array = eval(sys.argv[1])
    hps = array[int(sys.argv[2])]
    os.makedirs(hps['log_dir'], exist_ok=True)

    if '{task}' == 'qm9':
        trial = QM9MixtureModelTrainer(hps)
    elif '{task}' == 'seh':
        trial = SEHDoubleModelTrainer(hps)
    elif '{task}' == 'rna':
        trial = RNABindDoubleTrainer(hps)
    elif '{task}' == 'bitseq':
        trial = BitSeqMixTrainer(hps)
        
    trial.print_every = 1
    trial.verbose = True
    trial.run()
"""


def make_sh_script(run_name: str, log_dir: str, task: str, CUR_DIR):
    
    if task == 'rna':
        with open(f"{CUR_DIR}/utils/template_rna.sh", 'r') as f:
            script = f.read()
            f.close()
    else: 
        with open(f"{CUR_DIR}/utils/template.sh", 'r') as f:
            script = f.read()
            f.close()
    
    script = script.format(run_name, log_dir)
    return script

class RunObject():
    hparams: HParams
    log_dir: str
    run_name: str
    num_seeds: int
    task: str

    def __init__(
            self,
            task: str="seh",
            p: tuple=None,
            run_name: str=None,
            num_seeds: int=5,
            from_config: bool=False,
            config_path: str=None,
            LOG_ROOT: str=None,
            ground_truth: str=None,
        ):
        if from_config:
            with open(f"{config_path}/run_object.json", 'r') as f:
                config = json.load(f)
                f.close()
            self.__dict__ = config
            print(f"Loaded run object from {config_path}.\n")
        else:
            assert task in ["seh", "qm9", "bitseq", "rna"], "Task must be one of 'seh', 'qm9', 'rna' or 'bitseq'."
            self.task = task
            self.make_hparams_obj(p)
            self.num_seeds = num_seeds
            self.run_name = run_name if run_name else str(uuid4())
            self.log_dir = self.make_log_dir(LOG_ROOT=LOG_ROOT)
            self.ground_truth = ground_truth

    def make_hparams_obj(self, p):
        temp, use_replay, tb_variant, ddqn_update_step, buffer_size,\
            dqn_tau, dqn_epsilon, num_workers, num_training_steps, batch_size,\
            dqn_n_step, prob, p_greedy_sample, p_of_max_sample, p_quantile_sample = p
        
        self.hparams = {
            "temperature": temp if type(temp) == list else [temp],
            "use_replay": bool(use_replay),
            "tb_variant": str(tb_variant),
            "ddqn_update_step": int(ddqn_update_step),
            "buffer_size": int(buffer_size),
            "dqn_tau": float(dqn_tau),
            "dqn_epsilon": float(dqn_epsilon),
            "num_workers": int(num_workers),
            "num_training_steps": int(num_training_steps),
            "batch_size": int(batch_size),
            "dqn_n_step": int(dqn_n_step),
            "p": float(prob),
            "p_greedy_sample": bool(p_greedy_sample),
            "p_of_max_sample": bool(p_of_max_sample),
            "p_quantile_sample": bool(p_quantile_sample),
        }


    def make_final_params(self, BASE_HPS) -> Config:
        hps = {
            **BASE_HPS,
            'log_dir': self.log_dir,
            'num_workers': self.hparams['num_workers'],
            'num_training_steps': self.hparams['num_training_steps'],
            'cond': {
                **BASE_HPS['cond'],
                'temperature': {
                    **BASE_HPS['cond']['temperature'],
                    'dist_params': self.hparams['temperature'],
                }
            },
            'algo': {
                **BASE_HPS['algo'],
                'global_batch_size': self.hparams["batch_size"],
                'ddqn_update_step': self.hparams['ddqn_update_step'],
                'rl_train_random_action_prob': self.hparams['dqn_epsilon'],
                'dqn_tau': self.hparams['dqn_tau'],
            },
        }

        if self.hparams['tb_variant'] != "NoTB":
            hps['algo']['tb'] = { "variant": self.hparams['tb_variant'] }
            
        if self.hparams['use_replay']:
            hps['replay'] = {
                "use": True,
                "capacity": self.hparams['buffer_size'],
                "warmup": 2_000,
            }
        
        return hps
    
    def make_log_dir(self, LOG_ROOT) -> str:
        cur_date = datetime.today().strftime('%Y-%m-%d')
        cur_time = datetime.today().strftime('%H-%M-%S')
        log_dir = f'{LOG_ROOT}/{cur_date}/{self.run_name}-{cur_time}'
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    def generate_scripts(self, save_run: bool=True, CUR_DIR: str=None, BASE_HPS: Config=None):
        print(f"Generating scripts for run {self.run_name}...\n")

        final_params = self.make_final_params(BASE_HPS=BASE_HPS)

        py_script = make_py_script(self.task, final_params, self.log_dir)
        sh_script = make_sh_script(self.run_name, self.log_dir, self.task, CUR_DIR=CUR_DIR)
        sh_cmd = f"sbatch --array=0-{self.num_seeds-1} run.sh config"

        with open(f'{self.log_dir}/run.py', 'w') as f:
            f.write(py_script)

        with open(f'{self.log_dir}/run.sh', 'w') as f:
            f.write(sh_script)

        with open(f'{self.log_dir}/config.json', 'w') as f:
            json.dump(final_params, f, indent=4)

        with open(f'{self.log_dir}/howto.txt', 'w') as f:
            f.write(sh_cmd)

        if save_run:
            with open(f'{self.log_dir}/run_object.json', 'w') as f:
                json.dump(self.__dict__, f, indent=4)

        print("Scripts generated successfully in the following directory:\n")
        print(self.log_dir)
        print("\n\n")

    def print_obj(self):
        print(f"Setting up run {self.run_name} with the following hyperparameters:\n")
        print(self.hparams)
        print()
    