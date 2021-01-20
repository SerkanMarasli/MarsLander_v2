import numpy as np

X = [10., 499.]

if X[0] >= 700:
    X[0] = 699
if X[0] <= 0:
    X[0] = 1
if X[1] >= 500:
    X[1] = 499
if X[1] <= 0:
    X[1] = 1

obs = (int(round(X[1])), int(round(X[0])))

print(obs)