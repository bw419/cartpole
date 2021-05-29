import random
from mpl_toolkits import mplot3d
from matplotlib import patches
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import visvis as vv
import autograd.numpy as np
from matplotlib.pyplot import ion, draw, Rectangle, Line2D
import matplotlib.pyplot as plt
import os 
import time
import scipy
import sobol_seq
import scipy
from scipy.signal import argrelextrema
# import scipy.optimize

plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(__file__) + "/figs")
