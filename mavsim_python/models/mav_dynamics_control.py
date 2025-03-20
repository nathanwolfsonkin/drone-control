"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import numpy as np
from models.mav_dynamics import MavDynamics as RigidBody
# load message types
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
import parameters.aerosonde_parameters as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler


class MavDynamics(RigidBody):
    def __init__(self, Ts):
        super().__init__(Ts)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=MsgDelta())
        # update the message class for the true state
        self._update_true_state()


    ###################################
    # public functions
    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)
        super()._rk4_step(forces_moments)
        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)
        # update the message class for the true state
        self._update_true_state()

    ###################################
    # private functions
    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[:3]
        gust = wind[3:]
        wind_body = quaternion_to_rotation(self._state[6:10]) @ steady_state + gust
        self._wind = quaternion_to_rotation(self._state[6:10]).T @ wind_body
        velocity_body = self._state[3:6].reshape((3,1))
        v_air_body = velocity_body - wind_body
        self._Va = np.linalg.norm(v_air_body)
        self._alpha = np.arctan2(v_air_body.item(2), v_air_body.item(0)) if v_air_body[0] != 0 else 0.0
        self._beta = np.arcsin(v_air_body.item(1) / self._Va) if self._Va != 0 else 0.0

    def _forces_moments(self, delta: MsgDelta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :MAV delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        
        delta_a = delta.aileron
        delta_e = delta.elevator
        delta_r = delta.rudder
        delta_t = delta.throttle
        
        # extract states (phi, theta, psi, p, q, r)
        north = self._state.item(0)
        east = self._state.item(1)
        down = self._state.item(2)
        u = self._state.item(3)
        v = self._state.item(4)
        w = self._state.item(5)
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)
        
        grav_force = MAV.mass * MAV.gravity * np.array([[2 * (e1*e3 - e2*e0)],
                                                        [2 * (e2*e3 + e1*e0)],
                                                        [e3**2 + e0**2 - e1**2 - e2**2],
                                                        [0],
                                                        [0],
                                                        [0]])


        Va = self._Va
        alpha = self._alpha
        beta = self._beta
        
        # Linear Implementation
        CL = MAV.C_L_0 + MAV.C_L_alpha * alpha
        CD = MAV.C_D_0 + MAV.C_D_alpha * alpha
        
        # compute Lift and Drag Forces (F_lift, F_drag)
        F_lift = .5 * MAV.rho * Va**2 * MAV.S_wing * (
            CL + 
            MAV.C_L_q * MAV.c * q / (2 * Va) + 
            MAV.C_L_delta_e * delta_e
        )
        
        F_drag = .5 * MAV.rho * Va**2 * MAV.S_wing * (
            CD + 
            MAV.C_D_q * MAV.c * q / (2 * Va) + 
            MAV.C_D_delta_e * delta_e
        )
        
        # propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(Va, delta.throttle)
        
        prop_force_moment = np.array([thrust_prop, 0, 0, -torque_prop, 0, 0]).reshape(6,1)

        # compute longitudinal forces in body frame (fx, fz)
        R = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha),  np.cos(alpha)]])
        
        fx_fz = R @ np.array([[-F_drag],
                              [-F_lift]])

        fx = fx_fz.item(0)
        fz = fx_fz.item(1)
        
        # compute lateral forces in body frame (fy)
        fy = 0.5 * MAV.rho * Va**2 * MAV.S_wing * (
            MAV.C_Y_0 +
            MAV.C_Y_beta * beta +
            MAV.C_Y_p * MAV.b * p / (2 * Va) +
            MAV.C_Y_r * MAV.b * r / (2 * Va) +
            MAV.C_Y_delta_a * delta_a +
            MAV.C_Y_delta_r * delta_r
        )
        
        # compute logitudinal torque in body frame (My)
        My = .5 * MAV.rho * (Va**2) * MAV.S_wing * MAV.c * (
            MAV.C_m_0 + 
            MAV.C_m_alpha*alpha + 
            MAV.C_m_q * MAV.c * q / (2 * Va) + 
            MAV.C_m_delta_e * delta_e
        )
        
        print(delta_r)
        # compute lateral torques in body frame (Mx, Mz)
        Mx = 0.5 * MAV.rho * Va**2 * MAV.S_wing * MAV.b * (
            MAV.C_ell_0 +
            MAV.C_ell_beta * beta +
            MAV.C_ell_p * MAV.b * p / (2 * Va) +
            MAV.C_ell_r * MAV.b * r / (2 * Va) +
            MAV.C_ell_delta_a * delta_a +
            MAV.C_ell_delta_r * delta_r
        )
        
        Mz = 0.5 * MAV.rho * Va**2 * MAV.S_wing * MAV.b * (
            MAV.C_n_0 +
            MAV.C_n_beta * beta +
            MAV.C_n_p * MAV.b * p / (2 * Va) +
            MAV.C_n_r * MAV.b * r / (2 * Va) +
            MAV.C_n_delta_a * delta_a +
            MAV.C_n_delta_r * delta_r
        )
        
        forces_moments = np.array([fx, fy, fz, Mx, My, Mz]).reshape(6,1) + grav_force + prop_force_moment
        
        return forces_moments

    def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller
        V_in = MAV.V_max * delta_t
        
        # Quadratic formula to solve for motor speed
        a = MAV.C_Q0 * MAV.rho * np.power(MAV.D_prop, 5) / ((2 * np.pi)**2)
        b = (MAV.C_Q1 * MAV.rho * np.power(MAV.D_prop, 4) / (2 * np.pi)) * Va + (MAV.KQ**2) / MAV.R_motor
        c = MAV.C_Q2 * MAV.rho * np.power(MAV.D_prop, 3) * Va**2 - MAV.KQ / (MAV.R_motor) * V_in + MAV.KQ * MAV.i0

        # Consider only positive root
        omega_op = ( -b + np.sqrt(b**2 - 4*a*c)) / (2. * a)
        
        # compute advance ratio
        J_op = 2 * np.pi * Va / (omega_op * MAV.D_prop)
    
        CT = MAV.C_T2 * J_op**2 + MAV.C_T1 * J_op + MAV.C_T0
        CQ = MAV.C_Q2 * J_op**2 + MAV.C_Q1 * J_op + MAV.C_Q0
        
        # add thrust and torque due to propeller
        n = omega_op / (2*np.pi)
        fx = MAV.rho * n**2 * np.power(MAV.D_prop, 4) * CT
        Mx = MAV.rho * n**2 * np.power(MAV.D_prop, 5) * CQ

        return fx, Mx


    def _update_true_state(self):
        # rewrite this function because we now have more information
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0
