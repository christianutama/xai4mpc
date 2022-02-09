#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import do_mpc
import pickle
import sys
sys.path.append('../')
import tensorflow as tf


""" Choose model """
m = '4d' # either '2d' or '4d'

if m == '2d':
    from template_model_2d import template_model
elif m == '4d':
    from template_model_4d import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator



"""
Get configured do-mpc modules:
"""
model = template_model()
mpc = tf.keras.models.load_model('./models/nn_controller.h5')
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)


"""
Set initial state
"""

x0_min = np.array([[20.5], [18.0], [18.0], [ 5000.0]])
x0_max = np.array([[22.5], [25.0], [25.0], [15000.0]])
x0 = np.random.uniform(x0_min,x0_max) # Values between +3 and +3 for all states
simulator.x0 = x0
estimator.x0 = x0

"""
Scaling
"""
# States
x_lb = np.array([[20.0, 15.0,  0.0,     0.0]])
x_ub = np.array([[23.0, 25.0, 50.0, 20000.0]])

# Control inputs
u_lb = np.array([[-1000, -1000]])
u_ub = np.array([[ 1000,  1000]])


"""
Setup graphic:
"""

plt.ion()
fig, ax, graphics = do_mpc.graphics.default_plot(simulator.data)


"""
Run MPC main loop:
"""

for k in range(10): #range(7*24):

    # get control input from neural network
    # scale states
    x0_scaled = (x0.T - x_lb) / (x_ub - x_lb)
    weather_data_dummy = np.ones((1, 50)) * 0.5 # NOTE: Add here the respective (scaled) weather data
    u0_scaled = mpc.predict(np.hstack([x0_scaled, weather_data_dummy]))
    u0_real = u0_scaled * (u_ub - u_lb) + u_lb           # scale to real values
    u0_sat = np.minimum(np.maximum(u0_real, u_lb), u_ub) # ensure admissible control inputs via saturation

    # Compute the real control input variables
    P_HVAC = u0_sat[0, 0]
    if P_HVAC > 0:
        P_heat = P_HVAC
        P_cool = 0.0
    else:
        P_heat = 0.0
        P_cool = -P_HVAC

    P_bat = u0_sat[0, 1]

    u_applied = np.vstack([P_heat, P_cool, P_bat])

    y_next = simulator.make_step(u_applied)
    x0 = estimator.make_step(y_next)

graphics.plot_results()
graphics.reset_axes()

input('moinsen!')