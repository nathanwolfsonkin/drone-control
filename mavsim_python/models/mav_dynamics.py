"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
        7/13/2023 - RWB
        1/17/2024 - RWB
"""
import numpy as np
# load message types
from message_types.msg_state import MsgState
import parameters.aerosonde_parameters as MAV
from tools.rotations import euler_to_quaternion, quaternion_to_euler, quaternion_to_rotation

class MavDynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.north0],  # (0)
                               [MAV.east0],   # (1)
                               [MAV.down0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0],    # (12)
                               [0],   # (13)
                               [0],   # (14)
                               ])
        # initialize true_state message
        self.true_state = MsgState()

    ###################################
    # public functions
    def update(self, forces_moments):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        self._rk4_step(forces_moments)
        # update the message class for the true state
        self._update_true_state()

    def external_set_state(self, new_state):
        self._state = new_state

    ###################################
    # private functions
    def _rk4_step(self, forces_moments):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._f(self._state[0:13], forces_moments)
        k2 = self._f(self._state[0:13] + time_step/2.*k1, forces_moments)
        k3 = self._f(self._state[0:13] + time_step/2.*k2, forces_moments)
        k4 = self._f(self._state[0:13] + time_step*k3, forces_moments)
        self._state[0:13] += (time_step/6) * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2 + e1**2 + e2**2 + e3**2)
        if normE > 1e-6:  # Avoid division by numbers close to zero
            self._state[6][0] /= normE
            self._state[7][0] /= normE
            self._state[8][0] /= normE
            self._state[9][0] /= normE
        else:
            print("Warning: Quaternion normalization skipped due to near-zero norm.")

    
    def _f(self, state: np.ndarray, forces_moments: np.ndarray):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        
        # Extract the States
        north = state.item(0)
        east = state.item(1)
        down = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)

        # Extract Forces/Moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        Mx = forces_moments.item(3)
        My = forces_moments.item(4)
        Mz = forces_moments.item(5)
        
        # Position Kinematics
        def position_kinematics():            
            A = np.array([
                [e1**2 + e0**2 - e2**2 - e3**2,     2 * (e1 * e2 - e3 * e0),            2 * (e1 * e3 + e2 * e0)],
                [2 * (e1 * e2 + e3 * e0),           e2**2 + e0**2 - e1**2 - e3**2,      2 * (e2 * e3 - e1 * e0)],
                [2 * (e1 * e3 - e2 * e0),           2 * (e2 * e3 + e1 * e0),            e3**2 + e0**2 - e1**2 - e2**2]
            ])
            det_A = np.linalg.det(A)
            if det_A != 0:
                A = A / det_A
            else:
                print("Warning: Singular matrix detected. Normalization skipped.")

            
            x = np.array([[u],
                          [v],
                          [w]]).reshape(3,1)
            
            out =  A @ x
            return out.item(0), out.item(1), out.item(2)
        
        # Position dynamics
        def position_dynamics():
            A = np.array([[r*v - q*w],
                          [p*w - r*u],
                          [q*u - p*v]]).reshape(3,1)
            
            force_vect = np.array([[fx],
                                   [fy],
                                   [fz]]).reshape(3,1)
            
            out = A + (1/MAV.mass) * force_vect
            return out.item(0), out.item(1), out.item(2)
        
        # Rotational Kinematics
        def rotational_kinematics():
            
            A = np.array([
                [0, -p, -q, -r],
                [p,  0,  r, -q],
                [q, -r,  0,  p],
                [r,  q, -p,  0]
            ])
            
            x = np.array([[e0],
                          [e1],
                          [e2],
                          [e3]]).reshape(4,1)
            
            out = .5 * A @ x
            return out.item(0), out.item(1), out.item(2), out.item(3)
            
        # Rotational Dynamics
        def rotational_dynamics():
            A = np.array([[MAV.gamma1 * p * q - MAV.gamma2 * q * r],
                          [MAV.gamma5 * p * r - MAV.gamma6 * (p**2 - r**2)],
                          [MAV.gamma7 * p * q - MAV.gamma1 * q * r]]).reshape(3,1)
            
            l = Mx
            m = My
            n = Mz
            
            B = np.array([[MAV.gamma3 * l + MAV.gamma4 * n],
                          [(1/MAV.Jy) * m],
                          [MAV.gamma4 * l + MAV.gamma8 * n]]).reshape(3,1)
            
            out = A + B
            return out.item(0), out.item(1), out.item(2)

        north_dot, east_dot, down_dot = position_kinematics()
        
        # Position Dynamics
        u_dot, v_dot, w_dot = position_dynamics()
        
        # rotational kinematics
        e0_dot, e1_dot, e2_dot, e3_dot = rotational_kinematics()

        # rotatonal dynamics
        p_dot, q_dot, r_dot = rotational_dynamics()

        # collect the derivative of the states
        x_dot = np.array([north_dot, east_dot, down_dot, u_dot, v_dot, w_dot, e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]).reshape(13,1)
        
        return x_dot

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = 0
        self.true_state.alpha = 0
        self.true_state.beta = 0
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = 0
        self.true_state.gamma = 0
        self.true_state.chi = 0
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = 0
        self.true_state.we = 0
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0