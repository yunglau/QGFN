"""
Class methods for plotting runs
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from enum import Enum
from typing import List
import editdistance
import pickle

from rdkit import Chem, DataStructs
from .runs import RunObject
from .loaders import sqlite_load, rna_sqlite_load


def try_to_load_df(path: str) -> pd.DataFrame:
    """Try to load a dataframe from the given path."""
    try:
        return pd.read_csv(path)
    except:
        return None


class PlotType(Enum):
    """An enum representing the different types of plots that can be generated."""
    NUMBER_OF_MODES = 'number_of_modes'
    AVERAGE_REWARD = 'average_reward'
    TOP_K_REWARD = 'top_k_reward'
    REWARD_DISTRIBUTION = 'reward_distribution'
    TOP_K_SIMILARITY = 'top_k_similarity'
    NUMBER_OF_MODES_AT_K = 'number_of_modes_at_k'
    AVERAGE_REWARD_AT_K = 'average_reward_at_k'


class PlottableRunObject(RunObject):
    """
    A Base RunObject that supports plotting.
    """
    name: str           # The name of the run, used for plotting.
    color: str          # The color to use when plotting the run.
    workers: [int]      # The workers to use when plotting the run.
    num_points: int     # The number of points to plot.
    total_points: int   # The total number of points in the raw data.

    def __init__(self, path: str, name: str, color: str, workers: [int]=[0]):
        super().__init__(from_config=True, config_path=path)
        self.name = name
        self.color = color
        self.workers = workers

        # print(self.hparams)

        if self.hparams['num_workers'] == 0:
            self.hparams['num_workers'] = 1
        
        for worker in workers:
            assert 0 <= worker and worker < self.hparams['num_workers'], (
                f'Invalid worker number: {worker}.'
            )
        
        self.total_points = self.hparams['num_training_steps'] * self.hparams['batch_size']
        self.num_points = self.total_points *\
            len(workers) // self.hparams['num_workers'] + self.hparams['batch_size']

    def verify_plot_type(self, plot_type: PlotType) -> bool:
        """Verify that the run supports the given plot type."""
        if plot_type not in [
            PlotType.NUMBER_OF_MODES,
            PlotType.AVERAGE_REWARD,
            PlotType.TOP_K_SIMILARITY,
            PlotType.NUMBER_OF_MODES_AT_K,
            PlotType.AVERAGE_REWARD_AT_K
        ]:
            return False
        
        # TODO => Add more verification logic to ensure that the run supports the plot type.
        return True

    def load_raw_data(self, sqlite_cols: List[str]):
        """
        Loads the raw run data from the log directory. Please override this method 
        in your child class depending on your data format. This implementation supports
        the molecules environment which reads data from a sqlite database using the 
        specified columns.

        Parameters:
            sqlite_cols (List[str]): The columns to load from the sqlite database.

        Returns:
            List[Dict[str, List[List[float]]]]: A list of dictionaries containing the 
                raw data for each seed. The keys of the dictionary are the column names
                and the values are lists of lists containing the data for each worker.
        """
        return [
            sqlite_load(f'{self.log_dir}/seed-{i+1}/train/', sqlite_cols, 1)
            for i in range(self.num_seeds)
        ]
    
    def get_average_reward(self, save_df: bool=False) -> pd.DataFrame:
        """
        Returns a pandas dataframe with the average reward for each worker at each step.
        """
        pass
    
    @staticmethod
    def is_new_mode(obj, modes, sim_threshold=0.7) -> bool:
        """
        Returns True if obj is a new mode, False otherwise.
        """
        pass

    def get_modes(self, min_reward: float=0.9, sim_threshold: float=0.7, save_df: bool=False) -> pd.DataFrame:
        """
        Returns a pandas dataframe with the number of modes with reward > min_reward and 
        similarity > sim_threshold for each worker at each step.
        """
        pass

    def get_top_k_reward(self):
        pass

    def get_top_k_similarity(self, top_k: int=1000):
        pass

    def get_reward_distribution(self):
        pass


class MolsPlottableRunObject(PlottableRunObject):
    """Plottable Run Object for Molecules Environment"""

    def __init__(self, path: str, name: str, color: str, workers: [int]=[0]):
        super().__init__(path, name, color, workers)

    def get_average_reward(self, save_df: bool = False) -> pd.DataFrame:
        """
        Returns a pandas dataframe with the average reward for each worker at each step.
        NOTE: This method only supports pulling from the first worker for now.
        """
        df = try_to_load_df(f'{self.log_dir}/avg_reward.csv')
        if df is not None:
            return df

        cols = [f'fr_{w}' for w in self.workers]
        raw_data = self.load_raw_data(cols)

        L_all = min([len(i) for i in raw_data[0]['fr_0']])
        fr_i_all = np.array([i[:L_all] for i in raw_data[0][f'fr_0']]).T.flatten()
        fr_pad_step = len(fr_i_all)

        data = {'step_percent': np.linspace(0, 100, fr_pad_step)}
        df = pd.DataFrame({ key: pd.Series(value) for key, value in data.items() })
            
        # Get the average reward per timestep for each seed
        for i in range(self.num_seeds):
            min_num_steps = min([len(i) for i in raw_data[i][f'fr_0']])
            fr_i = np.array([i[:min_num_steps] for i in raw_data[i][f'fr_0']]).T.flatten()
            fr_i_pad_step = len(fr_i)

            df_temp = pd.DataFrame({ 'step': np.linspace(0, 100, len(fr_i)), 'value': fr_i })
            df_temp = df_temp.ewm(alpha=0.005).mean()

            # Doesn't this next line just get overwritten each iteration?
            df['step'] = np.linspace(0, 100, fr_i_pad_step)
            df['value_' + str(i)] = df_temp['value']

        if save_df:
            df.to_csv(f'{self.log_dir}/avg_reward.csv')

        return df
    
    @staticmethod
    def is_new_mode(obj, modes, sim_threshold=0.7):
        """
        Returns True if obj is a new mode, False otherwise.
        """      
        mol_obj = Chem.MolFromSmiles(obj)
        obj_id = Chem.RDKFingerprint(mol_obj)

        if len(modes) == 0:
            return True, obj_id

        if obj_id is None:
            return False, None
        
        sim_scores = DataStructs.BulkTanimotoSimilarity(obj_id, modes)
        return all(s < sim_threshold for s in sim_scores), obj_id
    
    def get_modes(self, min_reward: float=0.9, sim_threshold: float=0.7, save_df: bool=False) -> pd.DataFrame:
        """
        Returns a pandas dataframe with the number of modes with reward > min_reward and 
        similarity > sim_threshold for each worker at each step.
        """
        df = try_to_load_df(f'{self.log_dir}/num_modes.csv')
        if df is not None:
            return df
        
        cols = [f'fr_{w}' for w in self.workers] + ['smi']
        raw_data = self.load_raw_data(cols)

        data = {'step_percent': np.linspace(0, 100, self.num_points)}
        df = pd.DataFrame({ key: pd.Series(value) for key, value in data.items() })

        for i in range(self.num_seeds):
            # This only takers into account the molecules generated by the first worker!
            rewards = raw_data[i]['fr_0'][0]
            smiles = raw_data[i]['smi'][0]
            
            modes = []      # Store unique mode ids
            num_modes = []  # Store the number of unique modes at each step

            for j in tqdm(range(1, len(smiles))):

                if rewards[j] < min_reward or smiles[j] is None:
                    num_modes.append(len(modes))
                    continue

                is_new_mode, obj_id = MolsPlottableRunObject.is_new_mode(\
                    smiles[j], modes, sim_threshold=sim_threshold)

                if is_new_mode:
                    modes.append(obj_id)
                
                num_modes.append(len(modes))

            df['value_' + str(i)] = [0] + num_modes
         
        if save_df:
            df.to_csv(f'{self.log_dir}/num_modes.csv')

        return df

    @staticmethod
    def compute_tanim_top_k(smis, Rs, k):
        sims = []
        top = [(-100, None)]
        for i in range(len(smis)):
            if Rs[i] > top[0][0]:
                top.append((Rs[i], Chem.RDKFingerprint(Chem.MolFromSmiles(smis[i]))))
                top = sorted(top, key=lambda x: x[0])[-k:]
            if i > 0 and not i % 1000:
                pairwise_sims = []
                fps = [i[1] for i in top if i[1] is not None]
                for j in range(len(fps)-1):
                    pairwise_sims += list(DataStructs.BulkTanimotoSimilarity(fps[j], fps[j+1:]))
                sims.append(np.mean(pairwise_sims))
        return sims
    

    def get_top_k_similarity(self, top_k=1000, save_df: bool=False) -> pd.DataFrame:
        """
        Returns a pandas dataframe with the number of modes with reward > min_reward and 
        similarity > sim_threshold for each worker at each step.
        """
        df = try_to_load_df(f'{self.log_dir}/top_k_similarity.csv')
        if df is not None:
            return df
        
        cols = [f'fr_{w}' for w in self.workers] + ['smi']
        raw_data = self.load_raw_data(cols)

        data = {'step_percent': np.linspace(0, 100, 80)}
        df = pd.DataFrame({ key: pd.Series(value) for key, value in data.items() })

        for i in range(self.num_seeds):
            # This only takers into account the molecules generated by the first worker!
            rewards = raw_data[i]['fr_0'][0]
            smiles = raw_data[i]['smi'][0]
            sims = MolsPlottableRunObject.compute_tanim_top_k(smiles, rewards, top_k)

            df['value_' + str(i)] = sims

        if save_df:
            df.to_csv(f'{self.log_dir}/top_k_similarity.csv')

        return df

import pickle

class RNAPlottableRunObject(PlottableRunObject):
    """Plottable Run Object for RNA Synthesis Environment"""

    def __init__(self, path: str, name: str, color: str, workers: [int]=[0]):
        super().__init__(path, name, color, workers)
        
    def load_raw_data(self):
        return [
            rna_sqlite_load(f'{self.log_dir}/seed-{i+1}/train/')
            for i in range(self.num_seeds)
        ]

    def get_average_reward(self, save_df: bool = False) -> pd.DataFrame:
        """
        Returns a pandas dataframe with the average reward for each worker at each step.
        NOTE: This method only supports pulling from the first worker for now.
        """
        df = try_to_load_df(f'{self.log_dir}/avg_reward.csv')
        if df is not None:
            return df
        
        raw_data = self.load_raw_data()

        data = {'step_percent': np.linspace(0, 100, int(320256/4))}
        df = pd.DataFrame({ key: pd.Series(value) for key, value in data.items() })
            
        # Get the average reward per timestep for each seed
        for i in range(self.num_seeds):

            df['step'] = np.linspace(0, 100, int(len(raw_data[i]['fr_0'])/4))
            df['value_' + str(i)] = raw_data[i]['fr_0'][:int(320256/4)]
            df = df.ewm(alpha=0.005).mean()

        if save_df:
            df.to_csv(f'{self.log_dir}/avg_reward.csv')

        return df
    
    @staticmethod
    def is_new_mode():
        """
        Returns a set of ground truth for the target transcription 
        """      
        # with open('results/rna/L14_RNA1/mode_info.pkl', 'rb') as f:
        #     ground_truth = pickle.load(f)
        
        with open('results/rna/peaks_B2L14RNA1+2.txt') as file:
            ground_truth = set(line.strip() for line in file.readlines())
        
        return set(ground_truth)
    
    def get_modes(self, min_reward, sim_threshold, save_df: bool=False) -> pd.DataFrame:
        """
        Returns a pandas dataframe with the number of modes with reward > min_reward and 
        similarity > sim_threshold for each worker at each step.
        """
        df = try_to_load_df(f'{self.log_dir}/num_modes.csv')
        if df is not None:
            return df
        
        raw_data = self.load_raw_data()

        data = {'step_percent': np.linspace(0, 100, self.num_points*4)}
        df = pd.DataFrame({ key: pd.Series(value) for key, value in data.items() })

        ground_truth = self.is_new_mode()
        
        for i in range(self.num_seeds):
            # all workers values
            df_seed = raw_data[i]
            match_counts = []
            already_found = []
            original_length = len(df_seed)
            print(len(list(set(df_seed['smi'].to_list()) & ground_truth)))
            # Iterate over the fixed range
            for step in range(original_length):
                if df_seed.iloc[step]['smi'] in ground_truth and df_seed.iloc[step]['smi'] not in already_found:
                    already_found.append(df_seed.iloc[step]['smi'])
                    match_counts.append(1)
                else:
                    match_counts.append(0)
            
            df['value_' + str(i)] = np.cumsum(match_counts)

        if save_df:
            df.to_csv(f'{self.log_dir}/num_modes.csv')

        return df


class QM9PlottableRunObject(PlottableRunObject):
    """Plottable runs object for QM9 environment"""
    
    def __init__(self, path: str, name: str, color: str, workers: [int]=[0]):
        super().__init__(path, name, color, workers)

    def get_average_reward(self, save_df: bool = False) -> pd.DataFrame:
        df = try_to_load_df(f'{self.log_dir}/avg_reward.csv')
        if df is not None:
            return df
        
        cols = [f'fr_{w}' for w in self.workers]
        raw_data = self.load_raw_data(cols)

        L_all = min([len(i) for i in raw_data[0]['fr_0']])
        fr_i_all = np.array([i[:L_all] for i in raw_data[0][f'fr_0']]).T.flatten()
        fr_pad_step = len(fr_i_all)

        data = {'step_percent': np.linspace(0, 100, fr_pad_step)}
        df = pd.DataFrame({ key: pd.Series(value) for key, value in data.items() })
            
        # Get the average reward per timestep for each seed
        for i in range(self.num_seeds):
            min_num_steps = min([len(i) for i in raw_data[i][f'fr_0']])
            fr_i = np.array([i[:min_num_steps] for i in raw_data[i][f'fr_0']]).T.flatten()
            fr_i_pad_step = len(fr_i)

            df_temp = pd.DataFrame({ 'step': np.linspace(0, 100, len(fr_i)), 'value': fr_i })
            df_temp = df_temp.ewm(alpha=0.005).mean()

            # Doesn't this next line just get overwritten each iteration?
            df['step'] = np.linspace(0, 100, fr_i_pad_step)
            df['value_' + str(i)] = df_temp['value']

        if save_df:
            df.to_csv(f'{self.log_dir}/avg_reward.csv')

        return df

    @staticmethod
    def is_new_mode(obj, modes, sim_threshold=0.75):
        mol_obj = Chem.MolFromSmiles(obj)
        obj_id = Chem.RDKFingerprint(mol_obj)

        if len(modes) == 0:
            return True, obj_id

        if obj_id is None:
            return False, None
        
        sim_scores = DataStructs.BulkTanimotoSimilarity(obj_id, modes)
        return all(s < sim_threshold for s in sim_scores), obj_id
    
    def get_modes(self, min_reward: float=1.0, sim_threshold: float=0.75, save_df: bool=False) -> pd.DataFrame:
        df = try_to_load_df(f'{self.log_dir}/num_modes.csv')
        if df is not None:
            return df
        
        cols = [f'fr_{w}' for w in self.workers] + ['smi']
        raw_data = self.load_raw_data(cols)

        data = {'step_percent': np.linspace(0, 100, self.num_points)}
        df = pd.DataFrame({ key: pd.Series(value) for key, value in data.items() })

        for i in range(self.num_seeds):
            # This only takers into account the molecules generated by the first worker!
            rewards = raw_data[i]['fr_0'][0]
            smiles = raw_data[i]['smi'][0]
            
            modes = []      # Store unique mode ids
            num_modes = []  # Store the number of unique modes at each step

            for j in tqdm(range(1, len(smiles))):

                if rewards[j] < min_reward or smiles[j] is None:
                    num_modes.append(len(modes))
                    continue

                is_new_mode, obj_id = MolsPlottableRunObject.is_new_mode(\
                    smiles[j], modes, sim_threshold=sim_threshold)

                if is_new_mode:
                    modes.append(obj_id)
                
                num_modes.append(len(modes))

            df['value_' + str(i)] = [0] + num_modes
         
        if save_df:
            df.to_csv(f'{self.log_dir}/num_modes.csv')

        return df
    

class BitSeqPlottableRunObject(PlottableRunObject):
    """Plottable runs object for Bit Sequences task"""

    def __init__(self, path: str, name: str, color: str, workers: [int]=[0]):
        super().__init__(path, name, color, workers)

        # Load the ground truth modes from pkl file
        try:
            with open('./data/modes.pkl', 'rb') as f:
                self.ground_truth = set(pickle.load(f))
        except:
            raise Exception('Could not load ground truth modes from ./data/modes.pkl')
        
        self.num_unique_modes = len(self.ground_truth)

    def get_average_reward(self, save_df: bool = False) -> pd.DataFrame:
        df = try_to_load_df(f'{self.log_dir}/avg_reward.csv')
        if df is not None:
            return df
        
        cols = [f'fr_{w}' for w in self.workers]
        raw_data = self.load_raw_data(cols)

        L_all = min([len(i) for i in raw_data[0]['fr_0']])
        fr_i_all = np.array([i[:L_all] for i in raw_data[0][f'fr_0']]).T.flatten()
        fr_pad_step = len(fr_i_all)

        data = {'step_percent': np.linspace(0, 100, fr_pad_step)}
        df = pd.DataFrame({ key: pd.Series(value) for key, value in data.items() })
            
        # Get the average reward per timestep for each seed
        for i in range(self.num_seeds):
            min_num_steps = min([len(i) for i in raw_data[i][f'fr_0']])
            fr_i = np.array([i[:min_num_steps] for i in raw_data[i][f'fr_0']]).T.flatten()
            fr_i_pad_step = len(fr_i)

            df_temp = pd.DataFrame({ 'step': np.linspace(0, 100, len(fr_i)), 'value': fr_i })
            df_temp = df_temp.ewm(alpha=0.005).mean()

            # Doesn't this next line just get overwritten each iteration?
            df['step'] = np.linspace(0, 100, fr_i_pad_step)
            df['value_' + str(i)] = df_temp['value'], self.num_points

        if save_df:
            df.to_csv(f'{self.log_dir}/avg_reward.csv')

        return df
    
    def is_new_mode(self, obj, delta=28):
        if len(self.ground_truth) == 0:
            return False
        
        is_new = False
        to_remove = set()
        for mode in self.ground_truth:
            if editdistance.eval(obj, mode) <= delta:
                is_new = True
                to_remove.add(mode)

        self.ground_truth -= to_remove
        return is_new

    def get_modes(self, min_reward, sim_threshold: float=28, save_df: bool=False) -> pd.DataFrame:
        df = try_to_load_df(f'{self.log_dir}/num_modes.csv')
        if df is not None:
            return df
        
        # this is the minimum reward required to pass the threshold and be considered a "mode"
        min_reward = np.exp(1-(sim_threshold/120))

        # get the data from the sql file
        cols = [f'fr_{w}' for w in self.workers] + ['smi']
        raw_data = self.load_raw_data(cols)

        data = {'step_percent': np.linspace(0, 100, self.num_points)}
        df = pd.DataFrame({ key: pd.Series(value) for key, value in data.items() })

        for i in range(self.num_seeds):
            # This only takers into account the molecules generated by the first worker!
            rewards = raw_data[i]['fr_0'][0]
            smiles = raw_data[i]['smi'][0]

            num_modes = []
            for j in tqdm(range(1, len(smiles))):
                if smiles[j] is not None and rewards[j] >= min_reward:
                    is_new_mode = MolsPlottableRunObject.is_new_mode(\
                        smiles[j], sim_threshold=sim_threshold)
                    if is_new_mode:
                        print("Found a new mode, ", smiles[j])
                num_modes.append(self.num_unique_modes-len(self.ground_truth))

            df['value_' + str(i)] = [0] + num_modes
        
        if save_df:
            df.to_csv(f'{self.log_dir}/num_modes.csv')

        return df