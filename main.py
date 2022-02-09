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
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)


"""
Set initial state
"""

x0_min = np.array([[20.5], [18.0], [18.0], [ 5000.0]])
x0_max = np.array([[22.5], [25.0], [25.0], [15000.0]])
x0 = np.random.uniform(x0_min,x0_max) # Values between +3 and +3 for all states
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

# Use initial state to set the initial guess.
mpc.set_initial_guess()

"""
Setup graphic:
"""

fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data)
plt.ion()

"""
Run MPC main loop:
"""

for k in range(7*24):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

    graphics.plot_results(t_ind=k)
    graphics.plot_predictions(t_ind=k)
    graphics.reset_axes()
    plt.show()
    plt.pause(0.01)

input('Press any key to exit.')
