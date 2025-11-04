"""
plotting.py

Functions for plotting data

Author: Elliot Pallister
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

"""

PLOTTING FUNCTIONS

"""

def plot_pop_mean(mu, bins):

  mu = np.squeeze(mu)

  fig, ax = plt.subplots(figsize=(6, 3.5))

  for unit_trace in mu:
    ax.plot(bins, unit_trace, color='k', alpha=0.2, lw=1.0, zorder=1)

  pop_mean = mu.mean(axis=0)
  ax.plot(bins, pop_mean, color='k', lw=2.0, zorder=2)

  ax.axvline(0, color='black', linestyle='--', linewidth=1)
  ax.set_xlabel("Time from stimulus onset (s)")
  ax.set_ylabel("z-score")
  ax.set_title("Population mean trace across stimulus presentations of im036_r")

  return fig, ax


def mean_variance_scatter(mu, std, evoked_window, bins):

  mu = np.squeeze(mu)
  std = np.squeeze(std)

  evoked_window_mask = np.searchsorted(bins, evoked_window)
  means = mu[:, evoked_window_mask].mean(axis=1)
  var = (std[:, evoked_window_mask].mean(axis=1) ** 2)
  print(means)
  print(var)

  return None


  



  

  