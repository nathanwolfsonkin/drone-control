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
aircraft.plot_list_dict['time'] = []

for i in range(n):
    u = aircraft._forces_moments(np.array([0, 0, 0, 0]))
    x = integrator.step(t, x, u)
    t = (i+1) * dt
    aircraft.plot_list_dict['time'].append(t)

intg.__doc__
plt.figure()
plt.title('CD vs Alpha')
plt.plot(aircraft.plot_list_dict['alpha'], aircraft.plot_list_dict['CD'])
plt.xlabel('Alpha (rad)')
plt.ylabel('CD')

plt.figure()
plt.title('CL vs Alpha')
plt.plot(aircraft.plot_list_dict['alpha'], aircraft.plot_list_dict['CL'])
plt.xlabel('Alpha (rad)')
plt.ylabel('CL')

plt.figure()
plt.title('Lift vs Speed')
plt.scatter(aircraft.plot_list_dict['speed'], aircraft.plot_list_dict['F_lift'])
plt.xlabel('Speed (m/s)')
plt.ylabel('Lift (N)')

plt.figure()
plt.title('Drag vs Speed')
plt.scatter(aircraft.plot_list_dict['speed'], aircraft.plot_list_dict['F_drag'])
plt.xlabel('Speed (m/s)')
plt.ylabel('Drag (N)')

plt.figure()
plt.title('Time vs Speed')
plt.plot(aircraft.plot_list_dict['time'], aircraft.plot_list_dict['speed'])
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')


plt.show()

