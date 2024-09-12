"""
Data processing scripts that are specific to the molecules environment. These functions
should be replaced depending on the gflownet framework and environment you are working with.
"""

import sqlite3
import os
import glob
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd


# def sqlite_load(root, columns, num_workers=8, upto=None, begin=0):
#     try:
#         bar = tqdm(smoothing=0)
#         values = defaultdict(lambda: [[] for i in range(num_workers)])
#         for i in range(num_workers):
#             con = sqlite3.connect(f'file:{root}generated_mols_{i}.db?mode=ro', uri=True, timeout=6)
#             cur = con.cursor()
#             cur.execute('pragma mmap_size = 134217728')
#             cur.execute('pragma cache_size = -1024000;')
#             r = cur.execute(f'select {",".join(columns)} from results where rowid >= {begin}')
#             n = 0
#             for j, row in enumerate(r):
#                 bar.update()
#                 for value, col_name in zip(row, columns):
#                     values[col_name][i].append(value)
#                 n += 1
#                 if upto is not None and n * num_workers > upto:
#                     break
#             con.close()
#         return values
#     finally:
#         bar.close()

def sqlite_load(root, columns, num_workers=8, upto=None, begin=0):
    # try:
    #     bar = tqdm(smoothing=0)
    #     values = defaultdict(lambda: [[] for i in range(num_workers)])
    #     for i in range(num_workers):
    #         con = sqlite3.connect(f'file:{root}generated_mols_{i}.db?mode=ro', uri=True, timeout=6)
    #         cur = con.cursor()
    #         cur.execute('pragma mmap_size = 134217728')
    #         cur.execute('pragma cache_size = -1024000;')
    #         r = cur.execute(f'select {",".join(columns)} from results where rowid >= {begin}')
    #         n = 0
    #         for j, row in enumerate(r):
    #             bar.update()
    #             for value, col_name in zip(row, columns):
    #                 values[col_name][i].append(value)
    #             n += 1
    #             if upto is not None and n * num_workers > upto:
    #                 break
    #         con.close()
    #     return values
    # finally:
    #     bar.close()
    db_files = sorted(glob.glob(os.path.join(root, "*.db")))

    dataframes = []  # List to hold all dataframes

    for db_path in db_files:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql("SELECT * FROM results", conn)
            # print(df)
            # quit()
            dataframes.append(df)

    # Ensure all dataframes have the same number of rows
    # min_length = min(len(df) for df in dataframes)
    # trimmed_dfs = [df.iloc[:min_length].to_numpy() for df in dataframes]

    # Horizontally stack the arrays

    # Convert the stacked array back to a DataFrame
    # The column names will be numeric, starting from 0
    stacked_df = pd.concat(dataframes, axis=0).sort_index().reset_index()

    # print(stacked_df)

    return stacked_df
        

def rna_sqlite_load(root):
    db_files = glob.glob(os.path.join(root, "*.db"))

    combined_data = []
    columns = None

    for db_path in db_files:
        conn = sqlite3.connect(db_path)
        
        try:
            df = pd.read_sql("SELECT * FROM results", conn)
            
            if columns is None:
                columns = df.columns.tolist()
            
            combined_data.append(df)
        except sqlite3.OperationalError as e:
            if "no such table: results" in str(e):
                print(f"The table 'results' does not exist in the database at path: {db_path}")
            else:
                raise e
        finally:
            conn.close()

    if not combined_data:
        return pd.DataFrame(columns=columns or [])
    
    return pd.concat(combined_data, ignore_index=True)
