#%%

#Above '#%%' allows me to create visualisation by creating a cell

#IMPORTING RELEVANT MODULES

import math

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randint
from numpy.linalg import norm

from ipywidgets import interactive

from matplotlib import rcParams

rcParams['figure.figsize'] = (10, 8)


#INITIALLISATION



def mars_surface():
    surfaceN = randint(5, 15)
    land = np.zeros((surfaceN, 2), dtype=int)
    
    # first ensure there's a flat landing site at least 1000m long
    landing_site = randint(1, surfaceN-1)
    land[landing_site, 0] = randint(20, 30)
    land[landing_site+1, 0] = min(land[landing_site, 0] + randint(7, 12), 49)
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

def plot_lander(land, landing_site, X, thrust=None, animate=False, step=10):
    if animate:
        def plot_frame(n=len(X)-1):
            ax = plot_surface(land, landing_site)
            ax.plot(X[:n, 0], X[:n, 1], 'b--')
            ax.plot(X[n, 0], X[n, 1], 'b^', ms=20)
            if thrust is not None:
                ax.plot([X[n, 0], X[n, 0] - 1*thrust[n, 0]],
                        [X[n, 1] - 1., X[n, 1] - 1. - 1*thrust[n, 1]], 
                       'r-', lw=10)
            #plt.savefig('plot2')           
        return interactive(plot_frame, n=(0, len(X), step))
    else:
        ax = plot_surface(land, landing_site) 
        ax.plot(X[:, 0], X[:, 1], 'b--')
        ax.plot(X[-1, 0], X[-1, 1], 'b^')
        return ax

np.random.seed(56) # seed random number generator for reproducible results
land, landing_site = mars_surface()
# plot_surface(land, landing_site);

def interpolate_surface(land, x):          # height at any given x
    i,  = np.argwhere(land[:, 0] < x)[-1] # segment containing x is [i, i+1]
    m = (land[i+1, 1] - land[i, 1])/(land[i+1, 0] - land[i, 0]) # gradient
    x1, y1 = land[i, :] # point on line with eqn. y - y1 = m(x - x1) 
    return m*(x - x1) + y1

'''
x = 2000 # try varying x and testing if function works as expected
y = interpolate_surface(land, x)
plot_surface(land, landing_site)
plt.plot(x, y, 'bo')
'''

def height(land, X):
    return X[1] - interpolate_surface(land, X[0])

'''
assert abs(height(land, [1, land[0, 1]])) < 100.0 # height when on surface left edge should be close to zero
assert abs(height(land, [6999, land[-1, 1]])) < 100.0 # height when on surface at right edge should be close to zero


_land, _landing_site = mars_surface()

def _height(_land, X):
    return X[1] - interpolate_surface(_land, X[0])

points = np.zeros((10, 2))
points[:, 0] = randint(0, 7000, size=10)
points[:, 1] = randint(0, 3000, size=10)
for i in range(10):
    assert abs(height(_land, points[i, :]) - _height(_land, points[i, :])) < 1e-6
'''


# SIMULATE



g = 3.711 # m/s^2, gravity on Mars

def simulate(X0, V0, land, landing_site, 
             fuel=np.inf, dt=0.1, Nstep=1000, 
             autopilot=None, print_interval=100):
    
    n = len(X0)       # number of degrees of freedom (2 here)
    X = X0.copy()     # current position
    V = V0.copy()     # current velocity
    Xs = np.zeros((Nstep, n)) # position history (trajectory) 
    Vs = np.zeros((Nstep, n)) # velocity history
    Es = np.zeros(Nstep) # history of errors
    Rs = np.zeros(Nstep) # history of angles
    thrust = np.zeros((Nstep, n)) # thrust history
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
            rotate, power, e = autopilot(Es, i, X, V, fuel, rotate, power)
            Es[i] = e
            Rs[i] = rotate
            assert abs(rotate) <= 90
            assert 0 <= power <= 5

            rotate_rad = rotate * np.pi / 180.0 # degrees to radians
            thrust[i, :] = power * np.array([np.sin(rotate_rad), 
                                             np.cos(rotate_rad)])
            if fuel <= 0: 
                if not fuel_warning_printed:
                    print("Fuel empty! Setting thrust to zero")
                    fuel_warning_printed = True
                thrust[i, :] = 0
            else:
                fuel -= power*dt
        
        A = np.array([0, -g]) + thrust[i, :] # acceleration
        V += A * dt                          # update velocities
        X += V * dt                          # update positions
        
        if i % print_interval == 0: 
            print(f"i={i:03d} X=[{X[0]:8.3f} {X[1]:8.3f}] V=[{V[0]:8.3f} {V[1]:8.3f}]"
                  f" thrust=[{thrust[i, 0]:8.3f} {thrust[i, 1]:8.3f}] fuel={fuel:8.3f}")
        
        # check for safe or crash landing
        if X[1] < interpolate_surface(land, X[0]):
            if not (land[landing_site, 0] <= X[0] and X[0] <= land[landing_site + 1, 0]):
                print("crash! did not land on flat ground!")
            elif abs(rotate) >= 5:
                print("crash! did not land in a vertical position (tilt angle = 0 degrees)")
            elif abs(V[1]) >= 20:
                print("crash! vertical speed must be limited (<40m/s in absolute value), got ", abs(V[1]))
            elif abs(V[0]) >= 10:
                print("crash! horizontal speed must be limited (<20m/s in absolute value), got ", abs(V[0]))
            else:
                print("safe landing - well done!")
                success = True
            Nstep = i
            break
    
    return Xs[:Nstep,:], Vs[:Nstep,:], thrust[:Nstep,:], success, Es[:Nstep], Rs[:Nstep]

'''
X0 = [(land[landing_site+1, 0] + land[landing_site, 0]) // 2, 3000]
V0 = [0.0, 0.0]
Xs, Vs, thrust, success = simulate(X0, V0, land, landing_site)
plot_lander(land, landing_site, Xs, thrust, animate=True, step=10)
'''

m = 100. # mass of lander in kg
dt = 0.1

'''
Nstep = int((1.0 / dt) + 1) # number of steps required for 1 second of simulated time
Xs, Vs, thrust, success = simulate(X0, V0, land, landing_site, dt=dt, Nstep=Nstep)

t = np.array([dt*i for i in range(Nstep) ])
V = np.array([ m*g*Xs[i, 1] for i in range(Nstep) ])
T = np.array([ 0.5*m*norm(Vs[i, :])**2 for i in range(Nstep) ])
E = T + V

fig, ax = plt.subplots()
ax.plot(t, abs(E - E[0])/E[0], label="Total energy drift")
ax.set_xlabel('Time / s')
ax.set_ylabel('Energy drift')
ax.legend()

assert t[0] == 0.0
assert t[-1] >= 1.0
assert len(t) == len(E) == Nstep
assert abs(E[-1] - E[0])/E[0] < 1e-3
'''

def dummy_autopilot(i, X, V, fuel, rotate, power):
   return (rotate, power) # do nothing

'''
X0 = [(land[landing_site+1, 0] + land[landing_site, 0]) // 2, 3000]
V0 = [0., 0.]
Xs, Vs, thrust, success = simulate(X0, V0, land, landing_site, dt=dt, autopilot=dummy_autopilot)
plot_lander(land, landing_site, Xs, thrust, animate=True, step=10)
'''


#AUTOPILOT

# for 2D:
# either look at mag of velocity and have separate error for rotation
# or look at look at Vx and Vy separately and look into the rotation as a function of these

def proportional_autopilot(Es, i, X, V, fuel, rotate, power):
    # Xs, Vs, dt, e_history as arguments
    c = 1 # target landing speed, m/s
    K_h = 0.001
    K_p = 1.2
    K_d = 0.04 #0.0209
    K_i = 0.1 #0.0409
    k_horizontal = 0.5
    horizontaltarget = 1
    h = height(land, X)

    #write about tuning the parameters and look at how to do it automatically

    dE = (Es[i] - Es[i-1])/dt # derivative component

    sumERROR = 0 # sum of the errors not including the first and last (for trap rule)
    for j in range(i):
        sumERROR += Es[j+1]

    eIntegral = 0.5*dt*(Es[0] + Es[i] + 2*(sumERROR)) # integral component using Trapezium Rule
    # scipy.integrate.quad (for integral)
    # numpy.trapz (for integral)

    #e_vertical = - (c + K_h * h  + V[1]))
    #e_horizontal = - (ch + K_h2 * abs(diff between current and landing pos) + V[0])

    #power = f(e_vetical, e_horizontal) e.g. sqrt eh^2 + ev^2
    #rotation = g(e_vertical, e_horizontal) e.g. arctan(e_v / e_h)
 
    xdiff = (X[0] - ((land[landing_site+1, 0] + land[landing_site, 0]) // 2))
    ydiff = X[1] - (land[landing_site, 1]) 



    e = - (c + K_h*h + V[1])
    e_h = - (V[0] + horizontaltarget + k_horizontal*xdiff)

    #e_history[i] = e
    # (e(t)-e(t-dt))/dt) 
    #de = de/dt = - (0 + d/dt(K_h * h) + d/dt(V[1])
    #           = K_h * d/dt(X[1]) + d/dt(V[1])
    #         ~= K_h * (Xs[i, 1] - Xs[i-1, 1])/dt + (Vs[i, 1] - Vs[i-1, 1])/dt
    #diagram of geometry
    Pout = K_p*e + K_i*eIntegral + K_d*dE
    power = min(max(Pout, 0.0), 5)
    angle = min(90,max(np.arctan2(e_h,e)*(180/np.pi),-90))
    rotate = angle
    # print out e_h/e and angle to see what's causing problem
    
    '''
    if X[0] < ((land[landing_site+1, 0] + land[landing_site, 0]) // 2):
        
        xdiff = (((land[landing_site+1, 0] + land[landing_site, 0]) // 2) - X[0])
        ydiff = X[1] - (land[landing_site, 1])
        # +500 to give 500m space between landing and rotating, therefore, it can land where rotation = 0
        rotateradians = np.arctan((xdiff/ydiff))
        rotate = (rotateradians * 180/np.pi)

        # look into atan2... arctan2... to do above
        # adjust above such that it takes into account gradient
        
        if V[0] >= 15:
            rotate = -30
        
        
    elif X[0] > ((land[landing_site+1, 0] + land[landing_site, 0]) // 2):
        
        xdiff = (X[0] - ((land[landing_site+1, 0] + land[landing_site, 0]) // 2))
        ydiff = X[1] - (land[landing_site, 1])
        # +500 to give 500m space between landing and rotating, therefore, it can land where rotation = 0
        rotateradians = np.arctan((xdiff/ydiff))
        rotate = -(rotateradians * 180/np.pi)

        # look into atan2... arctan2... to do above
        # adjust above such that it takes into account gradient
        # try to use arctan2 to simplify above
        
        if V[0] <= -15:
            rotate = 30
    '''

    if i % 10 == 0:
        print(f'e={e:8.3f} Pout={Pout:8.3f} power={power:8.3f} rotation={rotate:8.3f}')
    return (rotate, power, e)


## testing different constant values


X0 = [45, 50]
V0 = [0., 0.]
Xs, Vs, thrust, success, Es, Rs = simulate(X0, V0, land, landing_site, dt=0.1, Nstep=3000, 
                                   autopilot=proportional_autopilot, fuel=1000)
plot_lander(land, landing_site, Xs, thrust, animate=True, step=1)# %%


#print(Vs)
#versions of plots to compare different settings on same graph
#look into plotting PD and PI controllers
#plot fuel use over time


# %%
