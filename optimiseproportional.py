# %%

# IMPORTING RELEVANT MODULES

import math

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randint
from numpy.linalg import norm

from ipywidgets import interactive

from matplotlib import rcParams

rcParams['figure.figsize'] = (10, 8)


# INITIALLISATION


def mars_surface():
    surfaceN = randint(5, 15)
    land = np.zeros((surfaceN, 2), dtype=int)

    # first ensure there's a flat landing site at least 1000m long
    landing_site = randint(1, surfaceN-1)
    land[landing_site, 0] = randint(2000, 5000)
    land[landing_site+1, 0] = min(land[landing_site, 0] +
                                  randint(1000, 2000), 6999)
    land[landing_site+1, 1] = land[landing_site, 1] = randint(1, 1500)

    # fill in the rest of the terrain
    for i in range(landing_site):
        land[i, 0] = (land[landing_site, 0] / landing_site) * i
        land[i, 1] = randint(0, 1500)

    for i in range(landing_site + 2, surfaceN):
        land[i, 0] = (land[landing_site + 1, 0] +
                      (7000 - land[landing_site + 1, 0]) / len(land[landing_site + 2:]) *
                      (i - (landing_site + 1)))
        land[i, 1] = randint(0, 1500)

    # impose boundary conditions
    land[0, 0] = 0
    land[-1, 0] = 6999

    return land, landing_site


def plot_surface(land, landing_site):
    fig, ax = plt.subplots()
    ax.plot(land[:landing_site+1, 0], land[:landing_site+1, 1], 'k-')
    ax.plot(land[landing_site+1:, 0], land[landing_site+1:, 1], 'k-')
    ax.plot([land[landing_site, 0], land[landing_site+1, 0]],
            [land[landing_site, 1], land[landing_site+1, 1]], 'k--')
    ax.set_xlim(0, 7000)
    ax.set_ylim(0, 3000)
    return ax


def plot_lander(land, landing_site, X, thrust=None, animate=False, step=10):
    if animate:
        def plot_frame(n=len(X)-1):
            ax = plot_surface(land, landing_site)
            ax.plot(X[:n, 0], X[:n, 1], 'b--')
            ax.plot(X[n, 0], X[n, 1], 'b^', ms=20)
            if thrust is not None:
                ax.plot([X[n, 0], X[n, 0] - 100*thrust[n, 0]],
                        [X[n, 1] - 100., X[n, 1] - 100. - 100*thrust[n, 1]],
                        'r-', lw=10)
        return interactive(plot_frame, n=(0, len(X), step))
    else:
        ax = plot_surface(land, landing_site)
        ax.plot(X[:, 0], X[:, 1], 'b--')
        ax.plot(X[-1, 0], X[-1, 1], 'b^')
        return ax


land, landing_site = mars_surface()
# plot_surface(land, landing_site);


def interpolate_surface(land, x):          # height at any given x
    i,  = np.argwhere(land[:, 0] < x)[-1]  # segment containing x is [i, i+1]
    m = (land[i+1, 1] - land[i, 1])/(land[i+1, 0] - land[i, 0])  # gradient
    x1, y1 = land[i, :]  # point on line with eqn. y - y1 = m(x - x1)
    return m*(x - x1) + y1


def height(land, X):
    return X[1] - interpolate_surface(land, X[0])


# SIMULATE


g = 3.711  # m/s^2, gravity on Mars


def simulate(Kh, Kp, X0, V0, land, landing_site,
             fuel=np.inf, dt=0.1, Nstep=1000,
             autopilot=None, print_interval=100):

    n = len(X0)       # number of degrees of freedom (2 here)
    X = X0.copy()     # current position
    V = V0.copy()     # current velocity
    Xs = np.zeros((Nstep, n))  # position history (trajectory)
    Vs = np.zeros((Nstep, n))  # velocity history
    thrust = np.zeros((Nstep, n))  # thrust history
    success = False
    fuel_warning_printed = False
    rotate = 0           # degrees, initial angle
    power = 0            # m/s^2, initial thrust power

    for i in range(Nstep):
        Xs[i, :] = X     # Store positions
        Vs[i, :] = V     # Store velocities -

 # Vs[:i,:].sum(axis=0) would give [sum(vx), sum(vy)] up to step i
 # change in velocity would be (Vs[i, :] - Vs[i-1, :]) / dt

        if autopilot is not None:
            # call user-supplied function to set `rotate` and `power`
            rotate, power = autopilot(Kh, Kp, i, X, V, fuel, rotate, power)
            assert abs(rotate) <= 90
            assert 0 <= power <= 4

            rotate_rad = rotate * np.pi / 180.0  # degrees to radians
            thrust[i, :] = power * np.array([np.sin(rotate_rad),
                                             np.cos(rotate_rad)])
            if fuel <= 0:
                if not fuel_warning_printed:
                    print("Fuel empty! Setting thrust to zero")
                    fuel_warning_printed = True
                thrust[i, :] = 0
            else:
                fuel -= power*dt

        A = np.array([0, -g]) + thrust[i, :]  # acceleration
        V += A * dt                          # update velocities
        X += V * dt                          # update positions

        if i % print_interval == 0:
            print(f"i={i:03d} X=[{X[0]:8.3f} {X[1]:8.3f}] V=[{V[0]:8.3f} {V[1]:8.3f}]"
                  f" thrust=[{thrust[i, 0]:8.3f} {thrust[i, 1]:8.3f}] fuel={fuel:8.3f}")

        # check for safe or crash landing
        if X[1] < interpolate_surface(land, X[0]):
            if not (land[landing_site, 0] <= X[0] and X[0] <= land[landing_site + 1, 0]):
                print("crash! did not land on flat ground!")
            elif rotate != 0:
                print(
                    "crash! did not land in a vertical position (tilt angle = 0 degrees)")
            elif abs(V[1]) >= 40:
                print(
                    "crash! vertical speed must be limited (<40m/s in absolute value), got ", abs(V[1]))
            elif abs(V[0]) >= 20:
                print(
                    "crash! horizontal speed must be limited (<20m/s in absolute value), got ", abs(V[0]))
            else:
                print("safe landing - well done!")
                success = True
            Nstep = i
            break

    return Xs[:Nstep, :], Vs[:Nstep, :], thrust[:Nstep, :], success, fuel


def dummy_autopilot(i, X, V, fuel, rotate, power):
    return (rotate, power)  # do nothing


# AUTOPILOT

def proportional_autopilot(Kh, Kp, i, X, V, fuel, rotate, power):
    c = 10.0  # target landing speed, m/s
    K_h = Kh    # needs to be between 0.01 and 0.15
    K_p = Kp    # needs to be between 0.15 and 0.30
    h = height(land, X)
    e = - (c + K_h*h + V[1])
    Pout = K_p*e
    power = min(max(Pout, 0.0), 4.0)
    if i % 100 == 0:
        print(f'e={e:8.3f} Pout={Pout:8.3f} power={power:8.3f}')
    return (rotate, power)

# testing to 'optimise' K_h and K_p

np.random.seed(105)  # seed random number generator for reproducible results
land, landing_site = mars_surface()

#Below to get kh and kp


kplist = np.arange(0.01, 1, 0.01, dtype=float).tolist()
khlist = np.arange(0.001, 0.05, 0.01, dtype=float).tolist()

X0 = [(land[landing_site+1, 0] + land[landing_site, 0]) // 2, 3000]
V0 = [0., 0.]

output = np.zeros((len(khlist), len(kplist), 2))

for i, Kh in enumerate(khlist):
    for j, Kp in enumerate(kplist):
        Xs, Vs, thrust, success, fuel = simulate(Kh, Kp, X0, V0, land, landing_site, dt=0.1, Nstep=2000,
                                         autopilot=proportional_autopilot, fuel=200)
        if success == True:
            output[i, j, :] =  (Vs[-1,1], fuel)


best_index = output[:, :, 1].argmax()

x, y = np.meshgrid(kplist, khlist)
plt.contourf(x, y, output[:,:, 1])
plt.colorbar()

'''
best_kp, best_kh = np.unravel_index(best_index, output[:, :, 1].shape)
plt.plot(kplist[best_kp], khlist[best_kh], 'rx')
'''

'''
for i in outputList:
    outputFuel.append(i[3])

print(max(outputFuel))
print(outputFuel.index(max(outputFuel)))

print(outputList[38])

Kh = outputList[38][0]
Kp = outputList[38][1]


Xs, Vs, thrust, success, fuel = simulate(Kh, Kp, X0, V0, land, landing_site, dt=0.1, Nstep=2000,
                                         autopilot=proportional_autopilot, fuel=200)

plot_lander(land, landing_site, Xs, thrust, animate=True, step=10)
'''

# plot 2D of landing velocity and fuel for Kh and Kp.
# Field contour plot 
# %%
