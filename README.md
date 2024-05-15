# QGFN: Controllable Greediness with Action Values

## Description
This repository contains the code to run the experiments and visualize the results highlighted in the paper [QGFN: Controllable Greediness with Action Values](https://arxiv.org/abs/2402.05234).

## Overview
Our codebase builds on top of a fork of the public [recursion gflownet](https://github.com/recursionpharma/gflownet) repo which provides the environment setup to run the gflownet framework on graph domains. Our main edits to the forked repo are found in the following files
- [src/gflownet/algo/graph_sampling.py](src/gflownet/algo/graph_sampling.py) handles the logit mixing between the gflownet policy with the learned Q value estimates. Here are permalinks to the implementations of the 3 flavours of qgfn mixing:
    - [`p-greedy`](src/gflownet/algo/graph_sampling.py#L148)
    - [`p-of-max`](src/gflownet/algo/graph_sampling.py#L202)
    - [`p-quantile`](src/gflownet/algo/graph_sampling.py#L172)
- [src/gflownet/data/mix_iterator.py](src/gflownet/data/mix_iterator.py) constructs batches of data for training, handles rewards, and logs summary statistics about the sampled trajectories
- [src/gflownet/tasks/](src/gflownet/tasks) is a directory that contains the runnable trainer files for the various tasks on which we validate our method

## Setup
To setup the project, run the following commands to install the required base packages:

Then, install the gflownet package from local source:
```bash
pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu116.html 
pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
pip install --no-index torch-cluster -f https://data.pyg.org/whl/torch-1.13.1+cu116.html 

pip install -e . --find-links https://data.pyg.org/whl/torch-1.13.1+cu116.html 
# It may take a while to build wheels for `torch-cluster`, `torch-scatter` & `torch-sparse
```

or follow the guideline from the public [recursion gflownet](https://github.com/recursionpharma/gflownet) repo

## How to run 
After setting up the virtual environment, you can test out QGFN by running `python interactive_script.py`. [interactive_script.py](interactive_script.py) has three flags, indicating the three variants of QGFN. Setting one of them to true will run one of the variants. 
- p_greedy_sample
- p_of_max_sample
- p_quantile_sample

## Usage
The current project supports training a mixture policy from scratch and visualizing the results of one or multiple runs in the same plots.

### Training from scratch
Our setup is made to support jobs running with the Slurm workload manager. To train a mixture policy from scratch, you must first edit the `utils/template.sh` file to customize the Slurm executable script to your GPU environment. Next, to generate the executables for a given job, simply edit the hyperparameters in the main function of `gen.py` and run the file from the project root directory:

#### Example
First, select the [task] on which you'd like to train the network.
Next, set the hyperparameters in the `gen.py` file. The main hyperparameters of interest are as follows:

```python
BASE_HPS: Config = {
    "log_dir": 'path.to.log.directory',
    "num_training_steps": 10000,
    ...
    "cond": {
        "temperature": {
            "sample_dist": "constant",
            "dist_params": 32.0,        # Set reward exponent beta
        }
    },
    "algo": {
        "p_greedy_sample": True,        # Set to true to run p-greedy sampling
        "p_of_max_sample": False,       # Set to true to run p-of-max sampling
        "p_quantile_sample": False,     # Set to true to run p-quantile sampling
        "p": 0.6,                       # Set value of `p` to control greediness
        "dqn_n_step": 30,               # Set n-step returns for Q learning
        ...
    },
    "task": {
        "qm9": {
            "h5_path": "path.to.dataset/qm9.h5",                 # Set path to qm9 dataset
            "model_path": "path.to.model/mxmnet_gap_model.pt"    # Set path to mxmnet proxy model for qm9 task reward
        },
        "bitseq": {
            "variant": "prepend-append",                         # Set variant of bit sequence generation mode as `prepend-append` or `autoregressive`
            "modes_path": "path.to.ref.sequences/modes.pkl",     # Set path to the pre-determined set of modes for the task 
            ...
        }
    }
}
```

Finally, you may set a list of hyperparameters in the `main()` method of `gen.py` for grid search. To generate the runs, simply run the file:
```bash
python gen.py
```

For every hyperparameter combination you specified, this will generate a corresponding run folder at `jobs/<current_date>/<run_id-current_time>`. This folder will contain the following files:

- `run.sh`: the Slurm executable script for the job
- `run.py`: the main executable script for the job
- `howto.txt`: a text file containing the command to submit the job to slurm
- `config.json`: a json file containing the hyperparameters for the job
- `run_object.json`: a json file containing the class instance of the run object, which can be used to re-instantiate the run object for downstream analysis and plotting

To submit the job to slurm, simply run the command specified in `howto.txt` from the run config directory. For example, if the command is `sbatch --array=0-4 run.sh config`, then run the following:

```bash
cd jobs/<current_date>/<run_id-current_time>
sbatch --array=0-4 run.sh config
```

### Plotting results
This repository also supports a variety of plotting functions to visualize and compare the results across multiple runs. The main plotting script is `vis.py` and we currently support the following plot types:

- `AVERAGE_REWARD`: plots the average reward per episode over the number of sampled trajectories during training
- `NUMBER_OF_MODES`: plots the number of unique modes above a certain reward threshold found by the mixture policy over the course of training
- `TOP_K_REWARD`: plots the average reward for the top k trajectories in the run with highest overall reward
- `TOP_K_SIMILARITY`: plots the tanimoto similarity (or other similarity measure) between the top k trajectories in the run with highest overall reward
- `NUMBER_OF_MODES_AT_K`: plots the number of modes at the `k`th or last `k` trajectory(ies) in the run
- `AVERAGE_REWARD_AT_K`: plots the average reward at the `k`th or last `k` trajectory(ies) in the run

To produce a given plot, you need to provide the path to the config folders (read above) of the runs you want to compare in the `runs` list in the main function of `vis.py`, along with a name and color for each run. For example, if you want to compare the results of two runs with paths `jobs/2023-11-19/001-01-54-02` and `jobs/2023-12-09/002-23-23-29`, you could write:

```python
runs = [
    {
        'path': 'jobs/2023-11-19/001-01-54-02',
        'name': 'run1',
        'color': 'blue'
    },
    {
        'path': 'jobs/2023-12-09/002-23-23-29',
        'name': 'run2',
        'color': 'red'
    },
]
```

Then, you need to specify the plot type you want to produce in the `plot_type` variable. Available plot types are specified in the `PlotType` enum in `utils/plotting.py`. For example, if you want to produce an average reward plot, you would write:

```python
plot_type = PlotType.NUMBER_OF_MODES
```

Finally, you can override the default plot parameters in the `plot_params` dictionary. For example, if you want to change the title of the plot, you could write:

```python
plot_params = {
    **DEFAULT_SHARED_PARAMS,
    **DEFAULT_UNIQUE_PARAMS[plot_type],

    # Add your custom parameters here.
    'title': 'My custom title'
}
```

The full specification of modifiable parameters for each plot type can be found in the `DEFAULT_UNIQUE_PARAMS` and `DEFAULT_SHARED_PARAMS` dictionaries in `vis.py`.

Once you have specified the runs, plot type, and plot parameters, you must run the script from the project root directory, and your plot will be saved in the `results` folder under a unique identifier based on the current date and plot type.

```bash
python vis.py
```

#### Plotting for other task environments
The above script generation and plotting code has been written to support the standard molecules environment in the `gflownet` package. However, the plotting code can be easily adapted to support other environments by inhering from the `PlottableRunObject` class in `utils/plotting.py`. This class defines the interface for a run object that can be plotted, and characterizes a few simple methods to extract the relevant data from the runs into a universal format, which is then provided to the downstream `PlotConfig` class for standard plotting. The methods that need to be implemented are:

```python
def load_raw_data(self, sqlite_cols: List[str]):
    """
    Loads the raw run data from the runs. Please override this method 
    in your child class depending on your data format. This implementation supports
    the molecules environment which reads data from a sqlite database using the 
    specified columns.
    """
    pass

def get_average_reward(self, save_df: bool=False) -> pd.DataFrame:
    """Returns a pandas dataframe with the average reward for each worker at each step"""
    pass

@staticmethod
def is_new_mode(obj, modes, sim_threshold=0.7) -> bool:
    """Returns True if obj is a new mode, False otherwise"""
    pass

def get_modes(self, min_reward: float=0.9, sim_threshold: float=0.7, save_df: bool=False) -> pd.DataFrame:
    """Returns a pandas dataframe with the number of modes with reward > min_reward and 
    similarity > sim_threshold for each worker at each step"""
    pass

def get_top_k_reward(self):
    """Returns a pandas dataframe with the top k reward for each worker at each step"""
    pass

def get_top_k_similarity(self):
    pass

def get_reward_distribution(self):
    pass
```
