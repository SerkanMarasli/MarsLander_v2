#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import math
from numpy.random import randint
from numpy.linalg import norm
from ipywidgets import interactive
from matplotlib import rcParams

def mars_surface():
    surfaceN = randint(5, 8)
    land = np.zeros((surfaceN, 2), dtype=int)
    
    # first ensure there's a flat landing site at least 1000m long
    landing_site = randint(1, surfaceN-1)
    land[landing_site, 0] = randint(7, 14)
    land[landing_site+1, 0] = min(land[landing_site, 0] + randint(3, 6), 20)
    land[landing_site+1, 1] = land[landing_site, 1] = randint(1, 6)
    
    # fill in the rest of the terrain
    for i in range(landing_site):
        land[i, 0] = (land[landing_site, 0] / landing_site) * i
        land[i, 1] = randint(0, 6)
    
    for i in range(landing_site + 2, surfaceN):
        land[i, 0] = (land[landing_site + 1, 0] + 
                      (20 - land[landing_site + 1, 0]) / len(land[landing_site + 2:]) * 
                      (i - (landing_site + 1)))
        land[i, 1] = randint(0, 6)
    
    # impose boundary conditions
    land[0, 0] = 0
    land[-1, 0] = 20

    return land, landing_site

def plot_surface(land, landing_site):
    fig, ax = plt.subplots()
    ax.plot(land[:landing_site+1, 0], land[:landing_site+1, 1], 'k-')
    ax.plot(land[landing_site+1:, 0], land[landing_site+1:, 1], 'k-')
    ax.plot([land[landing_site, 0], land[landing_site+1, 0]], 
             [land[landing_site, 1], land[landing_site+1, 1]], 'k--')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    return ax

np.random.seed(56) # seed random number generator for reproducible results
land, landing_site = mars_surface()

plot_surface(land, landing_site)



def interpolate_surface(land, x):          # height at any given x
    newarray = []
    for pair in land:
        if pair[0] < x:
            newarray.append(pair)
    print(newarray[-1][0])

interpolate_surface(land, 12)



# %%
