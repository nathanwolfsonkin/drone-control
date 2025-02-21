import numpy as np
import matplotlib.pyplot as plt
import simulation.integrators as intg
from simulation.rigidbody import RigidBody

# Nonlinear state space form:
#  xdot = f(t, x, u), 
#   t: time
#   x: state vector
#   u: input
mass = 1 # kg
Jxx = 1 # kg*m^2
Jyy = 1 # kg*m^2
Jzz = 2 # kg*m^2
Jxz = 0 # kg*m^2

pn = 0 # m 
pe = 0 # m
pd = 0 # m
u = 0 # m/s?
v = 0 # m/s
w = 0 # m/s
phi = 0 # deg
theta = 0 # deg
psi = 0 # deg
p = 1 # deg/s
q = 0 # deg/s
r = 2 # deg/s

t = 0;  
dt = 0.1
n = 100
x = np.array([pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]).reshape(12,1)
u = np.array([0, 0, 0, 0, 0, 0]).reshape(6,1)

rigid_body = RigidBody(mass,Jxx,Jyy,Jzz,Jxz,x)

f = rigid_body.full_ode

integrator = intg.RungaKutta4(dt, f)

t_history = [0]
omega_history = [np.reshape(x[9:11],2).tolist()]
omega3_history = [r]
for i in range(n):
    
    x = integrator.step(t, x, u)
    t = (i+1) * dt

    t_history.append(t)
    omega_history.append(np.reshape(x[9:11],2).tolist())
    omega3_history.append(x[11].item())
    
#################   ANALTTICAL SOLUTION   ########################

w_10 = p
w_20 = q
w_30 = r

v = w_30 * (Jxx - Jzz) / Jxx
time = np.array(t_history)
w1 = w_10 * np.cos(v*(time-time[0])) + w_20*np.sin(v*(time-time[0]))
w2 = w_20 * np.cos(v*(time-time[0])) - w_10*np.sin(v*(time-time[0]))


##################################################################

intg.__doc__
plt.figure()
plt.subplot(3,1,1)
plt.title('Numerical Solution')
plt.plot(t_history, omega_history)
plt.legend(['w1','w2'])
plt.subplot(3,1,2)
plt.title('Analytical Solution')
plt.plot(t_history, w1.tolist())
plt.plot(t_history, w2.tolist())
plt.legend(['w1_ana', 'w2_ana'])
plt.subplot(3,1,3)
plt.plot(t_history, omega3_history)
plt.title('w3')
plt.legend(['w3'])
plt.show()
