#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

HM_EPISODES = 2000000
MOVE_PENALTY = 1
LOSE_PENALTY = 300
WIN_REWARD = 100
epsilon = 0.999
EPS_DECAY = 0.99999  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 10000  # how often to play through env visually.

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
    WinSectionX_lower = 100
    WinSectionX_upper = 150
    WinSectionY = 75
    return WinSectionX_lower, WinSectionX_upper, WinSectionY

def LoseSection():
    LoseSectionX_lower = 100
    LoseSectionX_upper = 150
    LoseSectionY = 75 - 1
    return LoseSectionX_lower, LoseSectionX_upper, LoseSectionY

def action(choice):
    '''
    9 total actions
    '''
    if choice == 0:
        rotateX = 0
        thrust = 0
    elif choice == 1:
        rotateX = -1
        thrust = 0
    elif choice == 2:
        rotateX = 1
        thrust = 0
    elif choice == 3:
        rotateX = 0
        thrust = 0.5
    elif choice == 4:
        rotateX = 0
        thrust = -0.5
    elif choice == 5:
        rotateX = 1
        thrust = 0.5
    elif choice == 6:
        rotateX = -1
        thrust = 0.5
    elif choice == 7:
        rotateX = 1
        thrust = -0.5
    elif choice == 8:
        rotateX = 1
        thrust = 0.5

    return rotateX, thrust

if start_q_table is None:
    q_table = -1*np.ones(shape=(250,250,9)) # inital start with -1

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []

for episode in range(HM_EPISODES):
    X = [10., 249.]
    V = [0., 0.]
    rotate = 0
    cumulativePower = 0
    win_lower, win_upper, win_y = WinSection()
    lose_lower, lose_upper, lose_y = LoseSection()

    dt = 0.5

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
    
    episode_reward = 0

    for i in range(250):

        A = np.array([0., float(-g)])

        if X[0] >= 250:
            X[0] = 249
        if X[0] <= 0:
            X[0] = 1
        if X[1] >= 250:
            X[1] = 249
        if X[1] <= 0:
            X[1] = 1

        obs = (int(round(X[1]))-1, int(round(X[0]))-1)

        if np.random.uniform(0,1) > epsilon:
            chosenaction = np.argmax(q_table[obs])
        else:
            chosenaction = np.random.randint(0, 9)

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

        RoundedX = [round(X[0]), round(X[1])]

        if RoundedX[0] >= 250:
            RoundedX[0] = 249
        if RoundedX[0] <= 0:
            RoundedX[0] = 1
        if RoundedX[1] >= 249:
            RoundedX[1] = 249
        if RoundedX[1] <= 0:
            RoundedX[1] = 1

        if (RoundedX[0] > win_lower and RoundedX[0] < win_upper) and (RoundedX[1] == win_y) \
            and (V[0] <= 20 and V[1] <= 40) and (rotate <= 10):
            reward == WIN_REWARD
        #elif (RoundedX[0] > win_lower and RoundedX[0] < win_upper) and (RoundedX[1] == win_y):
           # reward = WIN_REWARD/10
        elif (RoundedX[0] > win_lower and RoundedX[0] < win_upper) and (RoundedX[1] == lose_y):
            reward = -LOSE_PENALTY
        elif RoundedX[1] <= 5:
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

    if epsilon > 0.3:
        epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
    
# %%
