import numpy as np

psi = 2
the = 10
phi = 20

rot1 = np.array([[np.cos(psi), -np.sin(psi), 0],
                 [np.sin(psi),  np.cos(psi), 0],
                 [0, 0, 1]])

rot2 = np.array([[np.cos(the),  0, np.sin(the)],
                 [0,            1,           0],
                 [-np.sin(the), 0, np.cos(the)]
                 ])

rot3 = np.array([[1, 0,                      0],
                 [0, np.cos(psi), -np.sin(psi)],
                 [0, np.sin(psi),  np.cos(psi)]
                 ])

R01 = rot1 @ rot2 @ rot3

p_com_0 = np.array([[0],
                  [0],
                  [-10]])


p_com_bat_1 = np.array([[0.2],
                      [0],
                      [0]])


p_bat = p_com_0 + R01 @ p_com_bat_1

print(p_bat)

v_1 = np.array([[15],
                [1],
                [.5]])

v_0 = R01 @ v_1

print('\n', v_0)

flight_path_angle= np.degrees(np.arctan2(v_0[2], v_0[0]))

print('\n', flight_path_angle)