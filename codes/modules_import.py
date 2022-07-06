'''
importing modules
@hatsyim
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import datetime
import random
import xarray as xr
import cartopy.crs as ccrs
import netCDF4
import matplotlib.pylab as pylab
import matplotlib.colors as mcolors
import h5py
import argparse
import pymap3d as pm
import os
import random
import time
import math
import torch
import copy
import re
import matplotlib.patches as patches
import cartopy.feature as cfeature

from scipy import signal
from scipy import spatial
from torch.nn import Linear
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from pyproj import Proj
from scipy.ndimage.filters import gaussian_filter
from glob import glob
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset
from mpl_toolkits.axes_grid1 import ImageGrid

# for reproducibility
os.environ['PYTHONHASHSEED']= '123'
np.random.seed(123)

# params = {
#     'xtick.labelsize':'xx-small',
#     'ytick.labelsize':'xx-small',
#     'figure.dpi':300
# }
# pylab.rcParams.update(params)

plt.style.use("./science.mplstyle")
