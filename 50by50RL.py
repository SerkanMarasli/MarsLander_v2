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

style.use("ggplot")

HM_EPISODES = 1000000
MOVE_PENALTY = 1
LOSE_PENALTY = 100
WIN_REWARD = 1000
epsilon = 0.99999
EPS_DECAY = 0.9999  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 5000   # how often to play through env visually.

LEARNING_RATE = 0.01
DISCOUNT = 0.95

start_q_table = None

g = 3.711 # need to change back

def plot_lander(land, landing_site, X):
    ax = plot_surface(land, landing_site) 
    ax.plot(X[:, 0], X[:, 1], 'b--')
    ax.plot(X[-1, 0], X[-1, 1], 'b^')
    return ax

def mars_surface():
    surfaceN = randint(5, 15)
    land = np.zeros((surfaceN, 2), dtype=int)
    
    # first ensure there's a flat landing site at least 1000m long
    landing_site = randint(1, surfaceN-1)
    land[landing_site, 0] = randint(20, 30)
    land[landing_site+1, 0] = min(land[landing_site, 0] + randint(7, 12), 50)
    land[landing_site+1, 1] = land[landing_site, 1] = randint(1, 15)
    
    # fill in the rest of the terrain
    for i in range(landing_site):
        land[i, 0] = (land[landing_site, 0] / landing_site) * i
        land[i, 1] = randint(0, 15)
    
    for i in range(landing_site + 2, surfaceN):
        land[i, 0] = (land[landing_site + 1, 0] + 
                      (50 - land[landing_site + 1, 0]) / len(land[landing_site + 2:]) * 
                      (i - (landing_site + 1)))
        land[i, 1] = randint(0, 15)
    
    # impose boundary conditions
    land[0, 0] = 0
    land[-1, 0] = 50

    return land, landing_site

def plot_surface(land, landing_site):
    fig, ax = plt.subplots()
    ax.plot(land[:landing_site+1, 0], land[:landing_site+1, 1], 'k-')
    ax.plot(land[landing_site+1:, 0], land[landing_site+1:, 1], 'k-')
    ax.plot([land[landing_site, 0], land[landing_site+1, 0]], 
             [land[landing_site, 1], land[landing_site+1, 1]], 'k--')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    return ax

np.random.seed(56) # seed random number generator for reproducible results
land, landing_site = mars_surface()
#plot_surface(land, landing_site)

def interpolate_surface(land, x):          # height at any given x
    i = len(np.argwhere(land[:, 0] < x)) - 1
    m = (land[i+1, 1] - land[i, 1])/(land[i+1, 0] - land[i, 0]) # gradient
    x1, y1 = land[i, :] # point on line with eqn. y - y1 = m(x - x1) 
    return m*(x - x1) + y1

def action(choice):
    '''
    9 total actions
    '''
    if choice == 0:
        rotateX = 0
        thrust = 0
    elif choice == 1:
        rotateX = -2.5
        thrust = 0
    elif choice == 2:
        rotateX = 2.5
        thrust = 0
    elif choice == 3:
        rotateX = 0
        thrust = 1
    elif choice == 4:
        rotateX = 0
        thrust = -1





    return rotateX, thrust


if start_q_table is None:
    q_table = -1*np.ones(shape=(50,50,5)) # inital start with -1
else:
    q_table = np.load("q_table-50by50.npy")

print(q_table)

episode_rewards = []
A1 = 0
A2 = 0

counter = 0
amount_of_wins = 0

for episode in range(HM_EPISODES):
    X = [5., 50.]
    V = [0., 0.]
    rotate = 0
    cumulativePower = 0
    Xhist = np.zeros((500,2))
    dt = 0.1
    

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
    
    episode_reward = 0

    for i in range(500):

        A = np.array([0., float(-g)])

        if X[0] >= 50:
            X[0] = 49
        if X[0] <= 0:
            X[0] = 1
        if X[1] >= 50:
            X[1] = 49
        if X[1] <= 0:
            X[1] = 1

        obs = (int(math.floor(X[1])), int(math.floor(X[0])))

        if np.random.uniform(0,1) > epsilon:
            chosenaction = np.argmax(q_table[obs])
        else:
            chosenaction = np.random.randint(0, 5)

        rotation, power = action(chosenaction)

        rotate += rotation

        if rotate >= 90:
            rotate = 90
        elif rotate <= -90:
            rotate = -90

        rotate_rad = rotate * np.pi / 180

        cumulativePower += power

        if cumulativePower >= 5:
            cumulativePower = 5
        elif cumulativePower <= 0:
            cumulativePower = 0

        A += cumulativePower*np.array([np.sin(rotate_rad), np.cos(rotate_rad)])
        V += A * dt
        X += V * dt

        if X[0] >= 50:
            X[0] = 49
        if X[0] <= 0:
            X[0] = 1
        if X[1] >= 50:
            X[1] = 49
        if X[1] <= 0:
            X[1] = 1

        RoundedX = [math.floor(X[0]), math.floor(X[1])]

        if RoundedX[0] >= 50:
            RoundedX[0] = 49
        if RoundedX[0] <= 0:
            RoundedX[0] = 1
        if RoundedX[1] >= 50:
            RoundedX[1] = 49
        if RoundedX[1] <= 0:
            RoundedX[1] = 1

        if X[1] < interpolate_surface(land, X[0]):
            if (land[landing_site, 0] <= X[0] and X[0] <= land[landing_site + 1, 0]) and \
                (abs(V[0]) <= 20 and abs(V[1] <= 40)) and (abs(rotate) <= 5):
                reward = WIN_REWARD
                A1 += 1
            #elif (land[landing_site, 0] <= X[0] and X[0] <= land[landing_site + 1, 0]):
                #reward = WIN_REWARD/10
                #A2 += 1
            else:
                reward = -LOSE_PENALTY
        else:
            reward = -MOVE_PENALTY


        new_obs = (RoundedX[1], RoundedX[0])

        max_future_q = np.max(q_table[new_obs])

        current_q = q_table[obs][chosenaction]

        #if reward == WIN_REWARD:
        #    new_q = WIN_REWARD
        #else:
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)

        q_table[obs][chosenaction] = new_q

        Xhist[i,:] = X

        episode_reward += reward
        if X[1] < interpolate_surface(land, X[0]):
            break
        
    
    # fix this 
    if episode % 20000 == 0:
        plot_lander(land, landing_site, Xhist[:i])
        plt.title(f'episode reward = {episode_reward}')
        plt.savefig(f'Figures/plotcount_{counter:05d}_ep_{episode:05d}.png')
        counter += 1

    episode_rewards.append(episode_reward)

    if epsilon > 0.001:
        epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

fig, ax = plt.subplots()
ax.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()
print(A1)
print(A2)

np.save("q_table-50by50", q_table)

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
# %%
