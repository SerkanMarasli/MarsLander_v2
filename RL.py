import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import time

{
    "python.linting.pylintArgs": [
        "--extension-pkg-whitelist=numpy"
    ]
}

style.use("ggplot")

SIZE = 100 # 100 by 100 grid
number_of_Episodes = 25000
move_Penalty = 1
fail_Penalty = 300
win_Reward = 100
epsilon = 0.9 # probability of taking random action = 1 - epsilon
epsilon_decay = 0.999
show_every = 3000 # show every 3000 episodes

start_q_table = None

learning_rate = 0.1
discount = 0.95

rocket_N = 1
win_N = 2
fail_N = 3

d = {1:(255,175,9), 2:(0,255,0), 3:(0,0,255)} # sets the colours

class Blocks:
    def __init__(self):
        self.x = np.random.randint(0,SIZE)
        self.y = np.random.randint(0,SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x), (self.y - other.y)

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1,2) # -1 up to 2
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1,2) # -1 up to 2
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1,y1), (x2,y2))] = [np.random.uniform(-5,0) for i in range(4)]

episode_rewards = []

for episode in range(number_of_Episodes):
    rocket = Blocks()
    fail = Blocks()
    win = Blocks()
    
    if episode % show_every == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{show_every} ep mean {np.mean(episode_rewards[-show_every:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (rocket-win, rocket-fail)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)

        rocket.action(action)

        if rocket.x == fail.x and rocket.y == fail.y:
            reward = -fail_Penalty
        elif rocket.x == win.x and rocket.x == win.y:
            reward = win_Reward
        else:
            reward = -move_Penalty

        new_obs = (rocket-win, rocket-fail)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == win_Reward:
            new_q = win_Reward
        elif reward == -fail_Penalty:
            new_q = -fail_Penalty
        else:
            new_q = (1-learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
        '''
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[win.y][win.x] = d[win_N]
            env[rocket.y][rocket.x] = d[rocket_N]
            env[fail.y][fail.x] = d[fail_N]

            img = Image.fromarray(env, "RGB")
            img = img.resize(300,300)
            cv2.imshow("", np.array(img))
            if reward == win_Reward or reward == -fail_Penalty:
                cv2.waitKey(2000)
                break
            else:
                cv2.waitKey(2)
        '''
        episode_reward += reward
        if reward == win_Reward or reward == -fail_Penalty:
            break

    episode_rewards.append(episode_reward)
    epsilon *= epsilon_decay

moving_avg = np.convolve(episode_rewards, np.ones((show_every,))/show_every, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {show_every}ma")
plt.xlabel("episode #")
plt.show()

