"""
Metrics for plotting and analysis
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

def aggregate_iqm(scores, axis):
  return stats.trim_mean(scores, proportiontocut=0.25, axis=axis)

def mean_confidence_interval(data, m, confidence=0.95):
  a = 1.0 * np.array(data)
  n = len(a)
  se = stats.sem(a)
  return m, m-se, m+se

def get_groupby_value(run, groupby):
  if groupby == 'dqn_n_step':
    groupby_value = run.hparams['dqn_n_step']
  elif groupby == 'beta':
    groupby_value = run.hparams['temperature'][0]
  elif groupby == 'p':
    groupby_value = run.hparams['p']
  else:
    raise ValueError('groupby must be one of: dqn_n_step, beta, p')
  return groupby_value

def smooth(x, n=100):
  # smoothing to reduce size of pdf 
  idx = np.int32(np.linspace(0, n-1e-3, len(x)))
  return np.linspace(0, len(x), n), np.bincount(idx, weights=x)/np.bincount(idx)

def smooth_ci(lo, hi, n=100):
  # smoothing to reduce size of pdf 
  assert len(lo) == len(hi)
  idx = np.int32(np.linspace(0, n-1e-3, len(lo)))
  return np.linspace(0, len(lo), n), np.bincount(idx, weights=lo)/np.bincount(idx), np.bincount(idx, weights=hi)/np.bincount(idx)