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

integrator = intg.Heun(dt, f)

heun_t_history = [0]
heun_x1_history = [x[0]]
heun_x2_history = [x[1]]
for i in range(n):
    
    x = integrator.step(t, x, u)
    t = (i+1) * dt

    heun_t_history.append(t)
    heun_x1_history.append(x[0])
    heun_x2_history.append(x[1])


x = np.array([1, 0])

integrator = intg.RungaKutta4(dt, f)

rk4_t_history = [0]
rk4_x1_history = [x[0]]
rk4_x2_history = [x[1]]
for i in range(n):
    
    x = integrator.step(t, x, u)
    t = (i+1) * dt

    rk4_t_history.append(t)
    rk4_x1_history.append(x[0])
    rk4_x2_history.append(x[1])

intg.__doc__
plt.figure()
plt.plot(rk4_t_history, rk4_x1_history)
plt.plot(heun_t_history, heun_x1_history)
plt.show()
