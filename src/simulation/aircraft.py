import numpy as np

from simulation.rigidbody import RigidBody
from simulation.wind_simulation import WindSimulation
import simulation.parameters as param

class Aircraft(RigidBody):
    def __init__(self, wind_conditions = WindSimulation().steady_state):
        super().__init__()
        self.wind_conditions = wind_conditions
        
        # Plotting history
        self.plot_list_dict = {
            'F_lift':[],
            'F_drag':[],
            'alpha':[],
            'speed':[]
        }
    
    def forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        
        delta_a, delta_e, delta_r, delta_t = delta.tolist()
        
        # extract states (phi, theta, psi, p, q, r)\
        phi = self.phi
        theta = self.theta
        psi = self.psi
        p = self.p
        q = self.q
        r = self.r

        # compute gravitational forces ([fg_x, fg_y, fg_z])
        grav_force = np.array([[-self.mass * param.gravity * np.sin(theta)],
                               [ self.mass * param.gravity * np.cos(theta) * np.sin(phi)],
                               [ self.mass * param.gravity * np.cos(theta) * np.cos(phi)],
                               [0],
                               [0],
                               [0]])

        # NON-LINEAR IMPLEMENTATION
        # compute Lift and Drag coefficients (CL, CD)
        u_r, v_r, w_r = (np.array([[self.u, self.v, self.r]]).T - self.wind_conditions).squeeze() # WIND VELOCITY VECTOR
        
        Va = np.linalg.norm(np.array([u_r, v_r, w_r]).T)
        
        alpha = np.arctan(w_r/u_r)
        alpha = alpha.item()
        
        beta = np.arcsin(v_r/Va)
        beta = beta.item()
        
        # CL = param.C_L_0 + param.C_L_alpha * alpha
        # CD = param.C_D_0 + param.C_D_alpha * alpha
        CL = param.C_L_0
        CD = param.C_D_0
        
        # compute Lift and Drag Forces (F_lift, F_drag)
        F_lift = .5 * param.rho * Va * param.S_wing * (
            param.C_L_0 + 
            param.C_L_alpha*alpha + 
            param.C_L_q * param.c * self.q / (2 * Va) + 
            param.C_L_delta_e * delta_e
        )
        
        F_drag = .5 * param.rho * Va * param.S_wing * (
            param.C_D_0 + 
            param.C_D_alpha*alpha + 
            param.C_D_q * param.c * self.q / (2 * Va) + 
            param.C_D_delta_e * delta_e
        )
        
        # propeller thrust and torque
        # thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta.throttle)

        # compute longitudinal forces in body frame (fx, fz)
        R = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha),  np.cos(alpha)]])
        
        fx, fz = (R @ np.array([[-F_drag],
                               [-F_lift]])).squeeze()

        # compute lateral forces in body frame (fy)
        fy = 0.5 * param.rho * Va**2 * param.S_wing * (
            param.C_Y_0 +
            param.C_Y_beta * beta +
            param.C_Y_p * param.b * p / (2 * Va) +
            param.C_Y_r * param.b * r / (2 * Va) +
            param.C_Y_delta_a * delta_a +
            param.C_Y_delta_r * delta_r
        )
        
        # compute logitudinal torque in body frame (My)
        My = .5 * param.rho * (Va**2) * param.S_wing * param.c * (
            param.C_m_0 + 
            param.C_m_alpha*alpha + 
            param.C_m_q * param.c * self.q / (2 * Va) + 
            param.C_m_delta_e * delta_e
        )
        
        # compute lateral torques in body frame (Mx, Mz)
        Mx = 0.5 * param.rho * Va**2 * param.S_wing * param.b * (
            param.C_ell_0 +
            param.C_ell_beta * beta +
            param.C_ell_p * param.b * p / (2 * Va) +
            param.C_ell_r * param.b * r / (2 * Va) +
            param.C_ell_delta_a * delta_a +
            param.C_ell_delta_r * delta_r
        )
        
        Mz = 0.5 * param.rho * Va**2 * param.S_wing * param.b * (
            param.C_n_0 +
            param.C_n_beta * beta +
            param.C_n_p * param.b * p / (2 * Va) +
            param.C_n_r * param.b * r / (2 * Va) +
            param.C_n_delta_a * delta_a +
            param.C_n_delta_r * delta_r
        )

        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T + grav_force
        
        
        # FOR PLOTTING PURPOSES
        self.plot_list_dict['F_lift'].append(F_lift)
        self.plot_list_dict['F_drag'].append(F_drag)
        self.plot_list_dict['alpha'].append(alpha)
        self.plot_list_dict['speed'].append(np.linalg.norm(np.sqrt(self.u**2 + self.v**2 + self.w**2)))
        return forces_moments