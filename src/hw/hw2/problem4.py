import numpy as np
import matplotlib.pyplot as plt
import simulation.integrators as intg
from simulation.rigidbody import RigidBody

# Nonlinear state space form:
#  xdot = f(t, x, u), 
#   t: time
#   x: state vector
#   u: input
mass = 11 # kg
Jxx = 0.824 # kg*m^2
Jyy = 1.135 # kg*m^2
Jzz = 1.759 # kg*m^2
Jxz = 0.120 # kg*m^2

pn = 0 # m 
pe = 0 # m
pd = -10 # m
u = 15 # m/s?
v = 1 # m/s
w = 0.5 # m/s
phi = 20 # deg
theta = 10 # deg
psi = 2 # deg
p = 0 # deg/s
q = 0 # deg/s
r = 0 # deg/s

t = 0;  
dt = 0.1; 
n = 100
x = np.array([pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]).reshape(12,1)
u = np.array([0, 0, 0, 0, 0, 0]).reshape(6,1)

rigid_body = RigidBody(mass,Jxx,Jyy,Jzz,Jxz,x)
f = rigid_body.full_ode

integrator = intg.Euler(dt, f)


t_history = [0]
x_history = [np.reshape(x,12).tolist()]
for i in range(n):
    
    x = integrator.step(t, x, u)
    t = (i+1) * dt

    t_history.append(t)
    x_history.append(np.reshape(x,12).tolist())

intg.__doc__
plt.figure()
plt.plot(t_history, x_history)
plt.show()
