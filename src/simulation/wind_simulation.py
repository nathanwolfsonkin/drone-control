"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""

import numpy as np


class WindSimulation:
    def __init__(self, steady_state = np.array([[0., 0., 0.]]).T):
        # steady state wind defined in the inertial frame
        self.steady_state = steady_state
