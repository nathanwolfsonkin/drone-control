import numpy as np
import matplotlib.pyplot as plt
import simulation.integrators as intg
from simulation.aircraft import Aircraft
import simulation.parameters as param

t = 0;  
dt = 0.1
n = 100

aircraft = Aircraft()

f = aircraft.full_ode

integrator = intg.RungaKutta4(dt, f)

# Initial conditions
x = param.state0.reshape(12,1)
u = aircraft.forces_moments(np.array([0, 0, 0, 0]))
aircraft.plot_list_dict['timelist'] = []

for i in range(n):
    
    x = integrator.step(t, x, u)
    t = (i+1) * dt
    u = aircraft.forces_moments(np.array([0, 0, 0, 0]))
    aircraft.plot_list_dict['timelist'].append(t)

intg.__doc__
plt.figure()
plt.title('Drag vs Alpha')
plt.plot(aircraft.plot_list_dict['alpha'], aircraft.plot_list_dict['F_drag'])
plt.xlabel('Alpha (rad)')
plt.ylabel('Drag (N)')
plt.show()
