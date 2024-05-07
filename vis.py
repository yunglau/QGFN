"""
This script generates standardized plots to evaluate and compare the performance of a set of 
runs. To use the script, simply specify the full directory paths to the runs you want to
compare, as well as the type of plot you want to generate. You can also provide some additional
parameters to customize the plots. The resulting plots will be saved under the results directory
with a unique date + id identifier.
"""

import os
import numpy as np
from uuid import uuid4
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
from tueplots import bundles
import seaborn as sns
from collections import defaultdict

from utils.metrics import (
    mean_confidence_interval,
    get_groupby_value,
    smooth,
    smooth_ci,
    aggregate_iqm
)

from utils.plotting import (
    PlotType,
    MolsPlottableRunObject,
    PlottableRunObject,
    RNAPlottableRunObject,
    QM9PlottableRunObject,
    BitSeqPlottableRunObject
)


# Load ICML2022 plot style into matplotlib global style
# plt_styles = bundles.icml2022()
# plt.rcParams.update(plt_styles)
# plt.rcParams.update(plt.rcParamsDefault)
sns.set(style="darkgrid")
# sns.set(font="Verdana")
sns.set_palette("pastel")
sns.set_theme()
plt.style.use('ggplot')
# print(f'Loaded plot style: {plt_styles}')


DEFAULT_SHARED_PARAMS = {
    'figsize': (5, 3),
    'grid': True,
    'legend': True,
    'save_dir': 'results',
    'save_format': 'pdf',
    'num_points_to_plot': 1000   # Number of points to plot
}


DEFAULT_UNIQUE_PARAMS = {
    PlotType.AVERAGE_REWARD: {
        'title': 'Average Reward',
        'xlabel': 'Trajectories Sampled',
        'ylabel': 'Average Reward',
        'confidence_interval': 0.95,
        'save_name': f'avg-reward-{datetime.now().strftime("%Y-%m-%d")}-{uuid4().hex}'
    },
    PlotType.NUMBER_OF_MODES: {
        'min_reward': 0.9,  # Don't forget to update ylabel when changing this!
        'sim_threshold': 0.7,
        'confidence_interval': 0.95,
        'title': 'Number of Modes',
        'xlabel': 'Trajectories Sampled',
        'ylabel': 'Number of Modes',
        'save_name': f'num-modes-{datetime.now().strftime("%Y-%m-%d")}-{uuid4().hex}'
    },
    PlotType.TOP_K_REWARD: {},
    PlotType.REWARD_DISTRIBUTION: {},
    PlotType.TOP_K_SIMILARITY: {
        'top_k': 1000,
        'title': 'Top-1000 Tanimoto similarity',
        'ylabel': 'Similarity score',
        'xlabel': 'Trajectories Sampled',
        'confidence_interval': 0.95,
        'save_name': f'top_k_similarity-{datetime.now().strftime("%Y-%m-%d")}-{uuid4().hex}'
    },
    PlotType.NUMBER_OF_MODES_AT_K: {
        'k': 60000,                 # Number of iterations at which to evaluate
        'is_last_k': False,         # If true, then averages over [-k:] instead at single k
        'groupby': 'dqn_n_step',    # Group x-axis by this parameter
        'title': 'Number of Modes at k',
        'ylabel': 'Number of Modes with',
        'xlabel': 'Some Parameter to Vary',
        'confidence_interval': 0.95,
        'min_reward': 0.9,
        'sim_threshold': 0.7,
        'save_name': f'num_modes_at_k-{datetime.now().strftime("%Y-%m-%d")}-{uuid4().hex}'
    },
    PlotType.AVERAGE_REWARD_AT_K: {
        'k': 60000,                 # Number of iterations at which to evaluate
        'is_last_k': False,         # If true, then averages over [-k:] instead at single k
        'groupby': 'dqn_n_step',    # Group x-axis by this parameter
        'title': 'Average Reward at k',
        'ylabel': 'Average Reward with',
        'xlabel': 'Some Parameter to Vary',
        'confidence_interval': 0.95,
        'save_name': f'avg_reward_at_k-{datetime.now().strftime("%Y-%m-%d")}-{uuid4().hex}'
    }
}


class PlotConfig():
    """A class representing the configuration for a plot."""
    plot_type: PlotType
    plot_params: Dict[str, any]
    runs: List[PlottableRunObject]
    required_shared_params: Dict[str, any] = {
        'figsize': tuple,
        'title': str,
        'xlabel': str,
        'ylabel': str,
        'grid': bool,
        'legend': bool,
        'save_dir': str,
        'save_name': str,
        'save_format': str,
        'num_points_to_plot': int
    }
    required_local_params: Dict[str, any] = {
        PlotType.NUMBER_OF_MODES: {
            'min_reward': float,
            'sim_threshold': float,
            'confidence_interval': float,
        },
        PlotType.AVERAGE_REWARD: {
            'confidence_interval': float,
        },
        PlotType.TOP_K_REWARD: {},
        PlotType.REWARD_DISTRIBUTION: {},
        PlotType.TOP_K_SIMILARITY: {
            'top_k': int,
        },
        PlotType.NUMBER_OF_MODES_AT_K: {
            'min_reward': float,
            'sim_threshold': float,
            'confidence_interval': float,
            'k': int,
            'is_last_k': bool,
            'groupby': str,
        },
        PlotType.AVERAGE_REWARD_AT_K: {
            'confidence_interval': float,
            'k': int,
            'is_last_k': bool,
            'groupby': str,
        }
    }

    def __init__(
            self,
            plot_type: PlotType,
            runs: List[PlottableRunObject],
            plot_params: Dict[str, any]
        ):
        self.runs = runs
        self.plot_type = plot_type
        self.plot_params = plot_params

        assert isinstance(plot_type, PlotType), f'Invalid plot type: {plot_type}'

        required_params = {
            **self.required_shared_params,
            **self.required_local_params[plot_type],
        }

        for param in required_params:
            assert param in plot_params, f'Missing shared parameter: {param}'
            assert isinstance(plot_params[param], required_params[param]), (
                f'Invalid type for parameter {param}: {type(plot_params[param])}.'
            )

    def prepare_plot(self):
        """Prepare the plot."""
        ax, fig = plt.subplots(figsize=self.plot_params['figsize'])
        plt.style.use('ggplot')
        fig.set_title(self.plot_params['title'], fontsize=11)
        fig.set_xlabel(self.plot_params['xlabel'], fontsize=9)
        fig.set_ylabel(self.plot_params['ylabel'], fontsize=9)
        fig.grid(self.plot_params['grid'])
        fig.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        sns.despine()
        ax.tight_layout()
        return ax, fig

    def prepare_multi_plot(self):
        """Prepare the plot."""
        # Change here: Set subplots to 1 row, 3 columns
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=self.plot_params['figsize'])
        plt.style.use('ggplot')

        # You now have an array of Axes objects in axs
        for ax in axs:
            ax.set_title(self.plot_params['title'], fontsize=11)
            ax.set_xlabel(self.plot_params['xlabel'], fontsize=9)
            ax.set_ylabel(self.plot_params['ylabel'], fontsize=9)
            ax.grid(self.plot_params['grid'])
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        sns.despine()
        fig.tight_layout()

        return fig, axs

    def plot(self):
        """Plot the data."""

        # Create the results directory if it doesn't exist
        if not os.path.exists(self.plot_params['save_dir']):
            os.makedirs(self.plot_params['save_dir'])

        ax, fig = self.prepare_plot()

        if self.plot_type == PlotType.NUMBER_OF_MODES:
            self.plot_number_of_modes(ax, fig)
        elif self.plot_type == PlotType.AVERAGE_REWARD:
            self.plot_average_reward(ax, fig)
        elif self.plot_type == PlotType.TOP_K_REWARD:
            raise NotImplementedError()
        elif self.plot_type == PlotType.REWARD_DISTRIBUTION:
            raise NotImplementedError()
        elif self.plot_type == PlotType.TOP_K_SIMILARITY:
            self.plot_top_k_similarity(ax, fig)
        elif self.plot_type == PlotType.NUMBER_OF_MODES_AT_K:
            self.plot_number_of_modes_at_k(ax, fig)
        elif self.plot_type == PlotType.AVERAGE_REWARD_AT_K:
            self.plot_average_reward_at_k(ax, fig)
        else:
            raise ValueError(f'Invalid plot type: {self.plot_type}')
        
        # Add legend if required
        if self.plot_params['legend']:
            fig.legend(frameon=False, fancybox=False, loc="best", fontsize='small')

        # Save the plot
        save_path = f'{self.plot_params["save_dir"]}/{self.plot_params["save_name"]}.{self.plot_params["save_format"]}'
        plt.savefig(save_path, format=self.plot_params['save_format'])

    def multi_plot(self, plot_types):
        """Plot the data."""

        # Create the results directory if it doesn't exist
        if not os.path.exists(self.plot_params['save_dir']):
            os.makedirs(self.plot_params['save_dir'])

        fig, axs = plt.subplots(nrows=1, ncols=len(plot_types), figsize=(5 * len(plot_types), 3))
        plt.style.use('ggplot')

        # Loop through each axis and corresponding plot type
        for ax, plot_type_each in zip(axs, plot_types):
            ax.set_title(plot_type_each.plot_params['title'], fontsize=11)
            ax.set_xlabel(plot_type_each.plot_params['xlabel'], fontsize=9)
            ax.set_ylabel(plot_type_each.plot_params['ylabel'], fontsize=9)
            ax.grid(plot_type_each.plot_params['grid'])
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

            if plot_type_each.plot_type == PlotType.NUMBER_OF_MODES:
                self.plot_number_of_modes(fig, ax)
            elif plot_type_each.plot_type == PlotType.AVERAGE_REWARD:
                self.plot_average_reward(fig, ax)
            elif plot_type_each.plot_type == PlotType.TOP_K_REWARD:
                raise NotImplementedError()
            elif plot_type_each.plot_type == PlotType.REWARD_DISTRIBUTION:
                raise NotImplementedError()
            elif plot_type_each.plot_type == PlotType.TOP_K_SIMILARITY:
                self.plot_top_k_similarity(fig, ax)
            else:
                raise ValueError(f'Invalid plot type: {plot_type_each.plot_type}')

            # Add legend if required
            if plot_type_each.plot_params.get('legend', False):
                ax.legend(frameon=False, fancybox=False, loc="best", fontsize='small')

        sns.despine()
        fig.tight_layout()

        # Save the plot
        save_path = f'{self.plot_params["save_dir"]}/{self.plot_params["save_name"]}.{self.plot_params["save_format"]}'
        plt.savefig(save_path, format=self.plot_params['save_format'])
        plt.close(fig)

    def plot_average_reward(self, ax: plt.Axes, fig: plt.Figure):
        dfs = [run.get_average_reward(save_df=True) for run in self.runs]
        conf_interval = self.plot_params['confidence_interval']
        n = self.plot_params['num_points_to_plot']

        for idx, df in enumerate(dfs):
            run = self.runs[idx]
            values = [df[f'value_{i}'] for i in range(run.num_seeds - 1)]
            y_range = aggregate_iqm(values, axis=0)

            fig.plot(*smooth(y_range, n=n), label=run.name, color=run.color)

            if conf_interval != None:
                _, ci_l, ci_h = mean_confidence_interval(values, y_range, conf_interval)
                fig.fill_between(*smooth_ci(ci_l, ci_h, n=n), alpha=0.4, facecolor=run.color)

    def plot_number_of_modes(self, ax: plt.Axes, fig: plt.Figure):
        dfs = [
            run.get_modes(
                min_reward=self.plot_params['min_reward'],
                sim_threshold=self.plot_params['sim_threshold'],
                save_df=True
            )
            for run in self.runs
        ]
        conf_interval = self.plot_params['confidence_interval']
        n = self.plot_params['num_points_to_plot']

        for idx, df in enumerate(dfs):
            run = self.runs[idx]
            # values = [df[f'value_{i}'] for i in range(5 - 1)]
            # y_range = aggregate_iqm(values, axis=0)

            # fig.plot(*smooth(y_range, n=n), label=run.name, color=run.color)
            
            for i in range(5):
                values = df[f'value_{i}']
                fig.plot(*smooth(values, n=n), label=f'seed {i}', color=run.color)

            # if conf_interval != None:
            #     _, ci_l, ci_h = mean_confidence_interval(values, y_range, conf_interval)
            #     fig.fill_between(*smooth_ci(ci_l, ci_h, n=n), alpha=0.4, facecolor=run.color)

    def plot_number_of_modes_at_k(self, ax: plt.Axes, fig: plt.Figure):
        dfs = [
            run.get_modes(
                min_reward=self.plot_params['min_reward'],
                sim_threshold=self.plot_params['sim_threshold'],
                save_df=True
            )
            for run in self.runs
        ]

        data = defaultdict(dict)   # {run_name: {groupby_value: [mode_values]}}
        colors = defaultdict(dict) # {run_name: color}

        k = self.plot_params['k']
        is_last_k = self.plot_params['is_last_k']
        groupby = self.plot_params['groupby']
        conf_interval = self.plot_params['confidence_interval']

        assert groupby in ['dqn_n_step', 'beta', 'p'], f'Invalid groupby parameter: {groupby}'

        for idx, df in enumerate(dfs):
            run = self.runs[idx]
            groupby_value = get_groupby_value(run, groupby)
            if is_last_k:
                data[run.name][groupby_value] =\
                    [np.mean(df[f'value_{i}'][-k:]) for i in range(run.num_seeds - 1)]
            else:
                data[run.name][groupby_value] = [df[f'value_{i}'][k] for i in range(run.num_seeds - 1)]
            colors[run.name] = run.color

        for run_name, modes in data.items():
            xs, ys = zip(*sorted(modes.items()))
            ys = np.array(ys).T
            y_range = mean_all(ys, axis=0)
            fig.plot(xs, y_range, color=colors[run_name], label=run_name)

            if conf_interval != None:
                _, ci_l, ci_h = mean_confidence_interval(ys, y_range, self.plot_params['confidence_interval'])
                fig.fill_between(xs, ci_l, ci_h, alpha=0.4, facecolor=colors[run_name])

    def plot_average_reward_at_k(self, ax: plt.Axes, fig: plt.Figure):
        dfs = [run.get_average_reward(save_df=True) for run in self.runs]

        data = defaultdict(dict)   # {run_name: {groupby_value: [mode_values]}}
        colors = defaultdict(dict) # {run_name: color}

        k = self.plot_params['k']
        is_last_k = self.plot_params['is_last_k']
        groupby = self.plot_params['groupby']
        conf_interval = self.plot_params['confidence_interval']

        assert groupby in ['dqn_n_step', 'beta', 'p'], f'Invalid groupby parameter: {groupby}'

        for idx, df in enumerate(dfs):
            run = self.runs[idx]
            groupby_value = get_groupby_value(run, groupby)
            if is_last_k:
                data[run.name][groupby_value] =\
                    [np.mean(df[f'value_{i}'][-k:]) for i in range(run.num_seeds - 1)]
            else:
                data[run.name][groupby_value] = [df[f'value_{i}'][k] for i in range(run.num_seeds - 1)]
            colors[run.name] = run.color

        for run_name, modes in data.items():
            xs, ys = zip(*sorted(modes.items()))
            ys = np.array(ys).T
            y_range = mean_all(ys, axis=0)
            fig.plot(xs, y_range, color=colors[run_name], label=run_name)

            if conf_interval != None:
                _, ci_l, ci_h = mean_confidence_interval(ys, y_range, self.plot_params['confidence_interval'])
                fig.fill_between(xs, ci_l, ci_h, alpha=0.4, facecolor=colors[run_name])

    def plot_top_k_reward(self):
        pass

    def plot_reward_distribution(self):
        pass

    def plot_top_k_similarity(self, ax: plt.Axes, fig: plt.Figure):
        dfs = [
            run.get_top_k_similarity(top_k=1000, save_df=True)
            for run in self.runs
        ]
        conf_interval = self.plot_params['confidence_interval']
        n = self.plot_params['num_points_to_plot']

        for idx, df in enumerate(dfs):
            run = self.runs[idx]
            values = [df[f'value_{i}'] for i in range(run.num_seeds - 1)]
            y_range = mean_all(values, axis=0)            
            fig.plot(*smooth(y_range), label=run.name, color=run.color)

            if conf_interval != None:
                _, ci_l, ci_h = mean_confidence_interval(values, y_range, conf_interval)
                fig.fill_between(*smooth_ci(ci_l, ci_h), alpha=0.4, facecolor=run.color)


if __name__ == "__main__":

    # Specify the runs you want to plot here.
    runs = [
        {
            'path': "/network/scratch/e/elaine.lau/logs/QGFN/2024-03-25/40081972-04f7-4c0b-a182-4cdaf1c52571-14-04-03/",
            'name': 'p-quantile QGFN',
            'color': 'red',
        },
    ]

    # Load the plottable run objects from the run paths above.        
    try:
        runs = [MolsPlottableRunObject(**run) for run in runs]
    except Exception as e:
        print(f'Error loading runs: {e}')
        exit(1)
    
    plot_type = PlotType.AVERAGE_REWARD
    # plot_type = PlotType.TOP_K_SIMILARITY
    plot_type = PlotType.NUMBER_OF_MODES
    # plot_type = PlotType.NUMBER_OF_MODES_AT_K
    # plot_type = PlotType.AVERAGE_REWARD_AT_K

    for run in runs:
        assert run.verify_plot_type(plot_type), (
            f'Run {run.name} does not support plot type {plot_type}.'
        )

    # Specify the plot parameters.
    plot_params = {
        **DEFAULT_SHARED_PARAMS,
        **DEFAULT_UNIQUE_PARAMS[plot_type],
        # Add your custom parameters here.
        # ...
        'min_reward': 0.97,
        'sim_threshold': 0.65,
    }

    # Generate and save the plot
    plotObject = PlotConfig(plot_type, runs, plot_params)

    # plotObject.multi_plot(plot_types)
    plotObject.plot()
