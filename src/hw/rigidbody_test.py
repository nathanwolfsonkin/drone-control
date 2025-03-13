import numpy as np
import matplotlib.pyplot as plt
from mavsim_python.models.mav_dynamics import MavDynamics as RigidBody
import mavsim_python.parameters.aerosonde_parameters as MAV
from tools.rotations import euler_to_quaternion

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
u = 0 # m/s
v = 0 # m/s
w = 0 # m/s
phi = 0 # deg
theta = 0 # deg
psi = 0 # deg
e = euler_to_quaternion(phi=phi, theta=theta, psi=psi)
p = 2.0 # deg/s
q = 0.0 # deg/s
r = 1.0 # deg/s

t = 0.0
dt = 0.1
n = 100
input = np.array([0, 0, 0, 0, 0, 0]).reshape(6,1)

state0 = np.array([pn, pe, pd, u, v, w, e.item(0), e.item(1), e.item(2), e.item(3), p, q, r]).reshape(13,1)

rigid_body = RigidBody(dt)
rigid_body.external_set_state(state0)

t_history = [0]
omega_history = [[p, q]]
omega3_history = [r]
for i in range(n):
    
    rigid_body.update(input)
    t = (i+1) * dt
    x = rigid_body._state
    
    t_history.append(t)
    omega_history.append([x.item(10), x.item(11)])
    omega3_history.append(x.item(12))
    
#################   ANALTTICAL SOLUTION   ########################

w_10 = p
w_20 = q
w_30 = r

v = w_30 * (Jxx - Jzz) / Jxx
time = np.array(t_history)
w1 = w_10 * np.cos(v*(time-time[0])) + w_20*np.sin(v*(time-time[0]))
w2 = w_20 * np.cos(v*(time-time[0])) - w_10*np.sin(v*(time-time[0]))


##################################################################

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
