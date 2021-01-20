#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

HM_EPISODES = 1000000
MOVE_PENALTY = 1
LOSE_PENALTY = 300
WIN_REWARD = 100
epsilon = 0.9
EPS_DECAY = 0.99999  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 1000  # how often to play through env visually.

LEARNING_RATE = 0.1
DISCOUNT = 0.95

ROCKET_N = 1  # player key in dict
WIN_CONDITION_N = 2  # food key in dict
LOSE_N = 3  # enemy key in dict

start_q_table = None # None or Filename

g = 3.711

# the dict!
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}

def WinSection():
    WinSectionX_lower = 300
    WinSectionX_upper = 500
    WinSectionY = 200
    return WinSectionX_lower, WinSectionX_upper, WinSectionY

def LoseSection():
    LoseSectionX_lower = 300
    LoseSectionX_upper = 500
    LoseSectionY = 200 - 1
    return LoseSectionX_lower, LoseSectionX_upper, LoseSectionY

def action(choice):
    '''
    9 total actions
    '''
    if choice == 0:
        rotate = 0
        thrust = 0
    elif choice == 1:
        rotate = -1
        thrust = 0
    elif choice == 2:
        rotate = 1
        thrust = 0
    elif choice == 3:
        rotate = 0
        thrust = 0.5
    elif choice == 4:
        rotate = 0
        thrust = -0.5
    elif choice == 5:
        rotate = 1
        thrust = 0.5
    elif choice == 6:
        rotate = -1
        thrust = 0.5
    elif choice == 7:
        rotate = 1
        thrust = -0.5
    elif choice == 8:
        rotate = 1
        thrust = 0.5

    return rotate, thrust

if start_q_table is None:
    q_table = np.zeros(shape=(500,700,9))

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []

for episode in range(HM_EPISODES):
    X = [10., 499.]
    V = [0., 0.]
    A = np.array([0., float(-g)])
    rotate = 0
    win_lower, win_upper, win_y = WinSection()
    lose_lower, lose_upper, lose_y = LoseSection()

    dt = 0.1

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
    
    episode_reward = 0

    for i in range(1000):

        if X[0] >= 700:
            X[0] = 699
        if X[0] <= 0:
            X[0] = 1
        if X[1] >= 500:
            X[1] = 499
        if X[1] <= 0:
            X[1] = 1

        obs = (int(round(X[1]))-1, int(round(X[0]))-1)

        if np.random.uniform(0,1) > epsilon:
            chosenaction = np.argmax(q_table[obs])
        else:
            chosenaction = np.random.randint(0, 9)

        rotation, power = action(chosenaction)

        rotate += rotation
        rotate_rad = rotate * np.pi / 180

        A += power*np.array([np.sin(rotate_rad), np.cos(rotate_rad)])
        V += A * dt
        X += V * dt

        RoundedX = [round(X[0]), round(X[1])]

        if RoundedX[0] >= 700:
            RoundedX[0] = 699
        if RoundedX[0] <= 0:
            RoundedX[0] = 1
        if RoundedX[1] >= 500:
            RoundedX[1] = 499
        if RoundedX[1] <= 0:
            RoundedX[1] = 1

        if (RoundedX[0] > win_lower and RoundedX[0] < win_upper) and (RoundedX[1] == win_y) \
            and (V[0] <= 20 and V[1] <= 40) and (rotate <= 10):
            reward == WIN_REWARD
        elif (RoundedX[0] > win_lower and RoundedX[0] < win_upper) and (RoundedX[1] == win_y):
            reward = WIN_REWARD/5
        elif (RoundedX[0] > win_lower and RoundedX[0] < win_upper) and (RoundedX[1] == lose_y):
            reward = -LOSE_PENALTY
        else:
            reward = -MOVE_PENALTY

        new_obs = (RoundedX[1], RoundedX[0])

        max_future_q = np.max(q_table[new_obs])

        current_q = q_table[obs][chosenaction]

        if reward == WIN_REWARD:
            new_q = WIN_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][chosenaction] = new_q

        episode_reward += reward
        if reward == WIN_REWARD or reward == -LOSE_PENALTY:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(epiosde_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

# %%
