import numpy as np
import matplotlib.pyplot as plt
import simulation.integrators as intg

# Nonlinear state space form:
#  xdot = f(t, x, u), 
#   t: time
#   x: state vector
#   u: input
def f(t, x, u):
    m = 1
    b = .25
    k = 1
    
    A = np.array([[0, 1],
                  [-k/m, -b/m]])
    
    xdot = A @ x
    
    return np.array([xdot[0], 
                     xdot[1]])

t = 0; x = np.array([1, 0]); u = 0
dt = 0.1; n = 100

integrator = intg.RungaKutta4(dt, f)

t_history = [0]
x_history = [x]
for i in range(n):
    
    x = integrator.step(t, x, u)
    t = (i+1) * dt

    t_history.append(t)
    x_history.append(x)

intg.__doc__
plt.figure()
plt.plot(t_history, x_history)
plt.show()
