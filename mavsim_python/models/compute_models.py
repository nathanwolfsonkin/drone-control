"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import numpy as np
from scipy.optimize import minimize
from tools.rotations import euler_to_quaternion, quaternion_to_euler
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
from message_types.msg_delta import MsgDelta
from models.mav_dynamics_control import MavDynamics


def compute_model(mav, trim_state, trim_input):
    # Note: this function alters the mav private variables
    A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file
    file = open('/Users/shreyas/workspaces/ME371_Fall2024/drone-control/mavsim_python/models/model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.elevator, trim_input.aileron, trim_input.rudder, trim_input.throttle))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
     A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
     A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
     A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
     A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    file.write('B_lon = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lon[0][0], B_lon[0][1],
     B_lon[1][0], B_lon[1][1],
     B_lon[2][0], B_lon[2][1],
     B_lon[3][0], B_lon[3][1],
     B_lon[4][0], B_lon[4][1],))
    file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
     A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
     A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
     A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
     A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    file.write('B_lat = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lat[0][0], B_lat[0][1],
     B_lat[1][0], B_lat[1][1],
     B_lat[2][0], B_lat[2][1],
     B_lat[3][0], B_lat[3][1],
     B_lat[4][0], B_lat[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.close()


def compute_tf_model(mav: MavDynamics, trim_state: np.ndarray, trim_input: MsgDelta):
    # trim values
    mav._state = trim_state
    mav._update_velocity_data()
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    phi, theta_trim, psi = quaternion_to_euler(trim_state[6:10])

    a_phi1 = -0.5 * MAV.rho * Va_trim**2 * MAV.S_wing * MAV.b * MAV.C_p_p * (MAV.b / (2 * Va_trim))
    a_phi2 = 0.5 * MAV.rho * Va_trim**2 * MAV.S_wing * MAV.b * MAV.C_p_delta_a

    a_theta1 = -MAV.rho * Va_trim**2 * MAV.c * MAV.S_wing * MAV.C_m_q * MAV.c / (2 * MAV.Jy * 2 * Va_trim)
    a_theta2 = -MAV.rho * Va_trim**2 * MAV.c * MAV.S_wing * MAV.C_m_alpha / (2 * MAV.Jy)
    a_theta3 = MAV.rho * Va_trim**2 * MAV.c * MAV.S_wing * MAV.C_m_delta_e / (2 * MAV.Jy)

    a_V1 = (MAV.rho*Va_trim*MAV.S_wing/MAV.mass)*(MAV.C_D_0 + MAV.C_D_alpha*alpha_trim + MAV.C_D_delta_e*trim_input.elevator) - (1/MAV.mass)*dT_ddelta_t(mav, Va_trim, trim_input.throttle)
    a_V2 = (1/MAV.mass)*dT_ddelta_t(mav, Va_trim, trim_input.throttle)
    a_V3 = MAV.gravity * np.cos(theta_trim - alpha_trim)

    return Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3


# Inputs: State with quaternion representation, Input as MsgDelta
def compute_ss_model(mav: MavDynamics, trim_state: np.ndarray, trim_input: MsgDelta):
    # Converts quaternion state to Euler state
    x_euler = euler_state(trim_state)

    # Compute Jacobians
    A = df_dx(mav, x_euler, trim_input)
    B = df_du(mav, x_euler, trim_input)
    
    # Extract longitudinal and lateral matrices
    A_lon = A[np.ix_([3, 5, 10, 7, 2], [3, 5, 10, 7, 2])]
    B_lon = B[np.ix_([3, 5, 10, 7, 2], [0, 3])]
    A_lat = A[np.ix_([4, 9, 11, 6, 8], [4, 9, 11, 6, 8])]
    B_lat = B[np.ix_([4, 9, 11, 6, 8], [1, 2])]
    return A_lon, B_lon, A_lat, B_lat


def euler_state(x_quat: np.ndarray) -> np.ndarray:
    x_euler = np.zeros((12, 1))
    x_euler[:6] = x_quat[:6]  # copy (pn, pe, h, u, v, w)
    euler_angles = quaternion_to_euler(x_quat[6:10])
    x_euler[6:9] = np.reshape(euler_angles, (3, 1))  # replace quaternion with Euler angles
    x_euler[9:] = x_quat[10:]  # copy (p, q, r)
    return x_euler


def quaternion_state(x_euler: np.ndarray) -> np.ndarray:
    x_quat = np.zeros((13, 1))
    x_quat[:6] = x_euler[:6]  # copy (pn, pe, h, u, v, w)
    quat = euler_to_quaternion(x_euler[6], x_euler[7], x_euler[8])
    x_quat[6:10] = quat.reshape((4, 1))  # replace Euler angles with quaternion
    x_quat[10:13] = x_euler[9:12]  # copy (p, q, r)
    return x_quat


def f_euler(mav: MavDynamics, x_euler, delta):
    #aymane helped me with this function
    x_quat = quaternion_state(x_euler)
    mav._state = x_quat
    mav._update_velocity_data()
    
    forces_moments = mav._forces_moments(delta)
    f_quat = mav._f(x_quat[0:13], forces_moments)

    # Convert quaternion derivative to Euler angle derivatives
    phi, theta, psi = quaternion_to_euler(x_quat[6:10])
    phi_dot = x_quat[10] + np.sin(phi) * np.tan(theta) * x_quat[11] + np.cos(phi) * np.tan(theta) * x_quat[12]
    theta_dot = np.cos(phi) * x_quat[11] - np.sin(phi) * x_quat[12]
    psi_dot = np.sin(phi)/np.cos(theta) * x_quat[11] + np.cos(phi)/np.cos(theta) * x_quat[12]

    f_euler_ = np.zeros((12, 1))
    f_euler_[0:3] = f_quat[0:3]       # pn_dot, pe_dot, pd_dot
    f_euler_[3:6] = f_quat[3:6]       # u_dot, v_dot, w_dot
    f_euler_[6] = phi_dot
    f_euler_[7] = theta_dot
    f_euler_[8] = psi_dot
    f_euler_[9:12] = f_quat[10:13]    # p_dot, q_dot, r_dot

    return f_euler_


def df_dx(mav: MavDynamics, x_euler: np.ndarray, delta: MsgDelta):
    A = np.zeros((12, 12))
    eps = 0.01
    for i in range(12):
        x_perturb = x_euler.copy()
        x_perturb[i, 0] += eps
        f1 = f_euler(mav, x_perturb, delta)
        f2 = f_euler(mav, x_euler, delta)
        A[:, i] = (f1 - f2).flatten() / eps
    return A


def df_du(mav, x_euler: np.ndarray, delta: MsgDelta):
    B = np.zeros((12, 4))
    eps = 0.01
    delta_array = np.array([
        delta.elevator,
        delta.aileron,
        delta.rudder,
        delta.throttle
    ])

    for i in range(4):
        delta_perturb = delta_array.copy()
        delta_perturb[i] += eps
        delta_perturb_MsgDelta = MsgDelta(elevator=delta_perturb[0], aileron=delta_perturb[1], rudder=delta_perturb[2], throttle=delta_perturb[3])
        f1 = f_euler(mav, x_euler, delta_perturb_MsgDelta)
        f2 = f_euler(mav, x_euler, delta)
        B[:, i] = (f1 - f2).flatten() / eps
    return B

def dT_dVa(mav: MavDynamics, Va, delta_t):
    eps = 0.01
    T1, Q1 = mav._motor_thrust_torque(Va, delta_t)
    T2, Q2 = mav._motor_thrust_torque(Va + eps, delta_t)
    dT_dVa = (T2 - T1) / eps
    return dT_dVa


def dT_ddelta_t(mav: MavDynamics, Va, delta_t):
    eps = 0.01
    T1, Q1 = mav._motor_thrust_torque(Va, delta_t)
    T2, Q2 = mav._motor_thrust_torque(Va, delta_t + eps)
    dT_ddelta_t = (T2 - T1) / eps
    return dT_ddelta_t