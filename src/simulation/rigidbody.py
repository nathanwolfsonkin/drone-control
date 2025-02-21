import numpy as np
import simulation.parameters as param

class RigidBody:
    def __init__(self, mass=param.mass, Jxx=param.Jx, Jyy=param.Jy, Jzz=param.Jz, Jxz=param.Jxz, x=param.state0):

        # Define mass/inertial properties
        self.mass = mass
        self.J = np.array([[Jxx,   0, Jxz],
                           [  0, Jyy,   0],
                           [Jxz,   0, Jzz]])
        
        # Define gamma constants
        self.gamma = Jxx*Jzz - Jxz**2
        self.gamma1 = Jxz*(Jxx - Jyy + Jzz) / self.gamma
        self.gamma2 = (Jzz*(Jzz - Jyy) + Jxz**2) / self.gamma
        self.gamma3 = Jzz / self.gamma
        self.gamma4 = Jxz / self.gamma
        self.gamma5 = (Jzz - Jxx) / Jyy
        self.gamma6 = Jxz / Jyy
        self.gamma7 = ((Jxx - Jyy) * Jxx + Jxz**2) / self.gamma
        self.gamma8 = Jxx / self.gamma
        
        # Define current state
        self.update_state(x)
        
    
    def first_ode(self):
        A = np.array([
        [
            np.cos(self.theta) * np.cos(self.psi), 
            np.sin(self.phi) * np.sin(self.theta) * np.cos(self.psi) - np.cos(self.phi) * np.sin(self.psi), 
            np.cos(self.phi) * np.sin(self.theta) * np.cos(self.psi) + np.sin(self.phi) * np.sin(self.psi)
        ],
        [
            np.cos(self.theta) * np.sin(self.psi), 
            np.sin(self.phi) * np.sin(self.theta) * np.sin(self.psi) + np.cos(self.phi) * np.cos(self.psi), 
            np.cos(self.phi) * np.sin(self.theta) * np.sin(self.psi) - np.sin(self.phi) * np.cos(self.psi)
        ],
        [
            -np.sin(self.theta), 
            np.sin(self.phi) * np.cos(self.theta),
            np.cos(self.phi) * np.cos(self.theta)
        ]
        ])
        x = np.array([[self.u],
                      [self.v],
                      [self.w]])
        
        return A @ x
    
    def second_ode(self, force_vect: np.ndarray):
        A = np.array([[self.r * self.v - self.q * self.w],
                      [self.p * self.w - self.r * self.u],
                      [self.q * self.u - self.p * self.v]])
        
        return A + (1/self.mass) * force_vect
    
    def third_ode(self):
        A = np.array([
            [1, np.sin(self.phi) * np.tan(self.theta), np.cos(self.phi) * np.tan(self.theta)],
            [0, np.cos(self.phi), -np.sin(self.phi)],
            [0, np.sin(self.phi) / np.cos(self.theta), np.cos(self.phi) / np.cos(self.theta)]
        ])
        x = np.array([[self.p],
                      [self.q],
                      [self.r]])
        
        return A @ x
        
    def fourth_ode(self, moment_vect):
        A = np.array([[self.gamma1 * self.p * self.q - self.gamma2 * self.q * self.r],
                      [self.gamma5 * self.p * self.r - self.gamma6 * (self.p**2 - self.r**2)],
                      [self.gamma7 * self.p * self.q - self.gamma1 * self.q * self.r]])
        
        l = moment_vect[0].item()
        m = moment_vect[1].item()
        n = moment_vect[2].item()
        
        B = np.array([[self.gamma3 * l + self.gamma4 * n],
                      [(1/self.J[1,1]) * m],
                      [self.gamma4 * l + self.gamma8 * n]])
        
        return A + B
        
    def full_ode(self, time, x, u):
        # Update State
        self.update_state(x)
        force_vect = u[:3]
        moment_vect = u[3:]
        
        # ODEs
        x1_3 = self.first_ode()
        x4_6 = self.second_ode(force_vect)
        x7_9 = self.third_ode()
        x10_12 = self.fourth_ode(moment_vect)
        return np.vstack((x1_3, x4_6, x7_9, x10_12)).reshape(12,1)

    def update_state(self, x):
        # Define current state
        self.pn = x[0].item()
        self.pe = x[1].item()
        self.pd = x[2].item()
        self.u = x[3].item()
        self.v = x[4].item()
        self.w = x[5].item()
        self.phi = x[6].item()
        self.theta = x[7].item()
        self.psi = x[8].item()
        self.p = x[9].item()
        self.q = x[10].item()
        self.r = x[11].item()