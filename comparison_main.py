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
plt.ion()
from casadi import *
from casadi.tools import *
import pdb
import sys
import do_mpc
import pickle
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

n_days = 2
init_offset = np.random.randint(2000 - (n_days + 1) * 24 - 1)
model = template_model()
mpc = template_mpc(model, init_offset)
nnc = tf.keras.models.load_model('./models/nn_controller_small.h5')
simulator_mpc = template_simulator(model, init_offset)
simulator_nnc = template_simulator(model, init_offset)
estimator_mpc = do_mpc.estimator.StateFeedback(model)
estimator_nnc = do_mpc.estimator.StateFeedback(model)


""" Data scaling """
x_lb = np.array([[20.0, 15.0,  0.0,     0.0]])
x_ub = np.array([[23.0, 25.0, 50.0, 20000.0]])
u_lb = np.array([[-1000, -1000]])
u_ub = np.array([[ 1000,  1000]])
p_lb = np.array([[-10.0,    0.0]])
p_ub = np.array([[ 30.0, 1200.0]])


""" Load disturbance data """
with open(r'./data/exttemp_and_solrad_2008.pkl', 'rb') as f:
    data = pickle.load(f)
T_data = data['T']
sr_data = data['sr']
P_data = np.hstack([T_data, sr_data])
P_data_s = (P_data - p_lb) / (p_ub - p_lb)
T_s = P_data_s[:,0].reshape(1, -1)
sr_s = P_data_s[:,1].reshape(1, -1)

"""
Set initial state
"""

x0_min = np.array([[20.5], [18.0], [18.0], [ 5000.0]])
x0_max = np.array([[22.5], [25.0], [25.0], [15000.0]])
x0 = np.random.uniform(x0_min, x0_max)
mpc.x0 = x0
n_h = mpc.n_horizon
simulator_mpc.x0 = x0
simulator_nnc.x0 = x0
estimator_mpc.x0 = x0
estimator_nnc.x0 = x0

x0_mpc = x0
x0_nnc = x0

# Use initial state to set the initial guess.
mpc.set_initial_guess()


"""
Run MPC main loop:
"""

for k in range(n_days * 24):

    # compute inputs
    u0_mpc = mpc.make_step(x0_mpc)

    # current position in disturbance data
    c = init_offset + k
    # scale current state
    x0_nnc_s = (np.reshape(x0_nnc, (1, -1)) - x_lb) / (x_ub - x_lb)
    # concatenate input to nnc
    inp = np.hstack([x0_nnc_s, T_s[:, c:c + n_h + 1], sr_s[:, c:c + n_h + 1]])
    # obtain scaled control input from nnc
    u0_nnc_s = nnc.predict(inp)
    # unscale control input from nnc
    u0_nnc = u0_nnc_s * (u_ub - u_lb) + u_lb

    # convert P_hvac to P_heat and P_cool and P_bat and input saturation
    u0_nnc_app = np.zeros((3,1))
    if u0_nnc[0,0] > 0:
        u0_nnc_app[0, 0] = np.minimum(u0_nnc[0,0], 1000.0)
        u0_nnc_app[1, 0] = 0.0
    elif u0_nnc[0,0] < 0:
        u0_nnc_app[0, 0] = 0.0
        u0_nnc_app[1, 0] = np.minimum(-u0_nnc[0, 0], 1000.0)
    u0_nnc_app[2, 0] = np.maximum(np.minimum(u0_nnc[0, 1], 1000.0), -1000.0)

    # simulate next step
    y_next_mpc = simulator_mpc.make_step(u0_mpc)
    y_next_nnc = simulator_nnc.make_step(u0_nnc_app)

    # estimator
    x0_mpc = estimator_mpc.make_step(y_next_mpc)
    x0_nnc = estimator_nnc.make_step(y_next_nnc)

fig, ax = plt.subplots(4, 1)
for label, simulator in zip(['mpc', 'nnc'], [simulator_mpc, simulator_nnc]):
    ax[0].plot(simulator.data['_x', 'T_r'], label = label)
    ax[1].plot(simulator.data['_x', 'E_bat'])
    ax[2].plot(simulator.data['_aux', 'P_hvac'])
    ax[3].plot(simulator.data['_u', 'P_bat'])
ax[0].legend()
ax[0].set_ylabel('T_r')
ax[1].set_ylabel('E_bat')
ax[2].set_ylabel('P_hvac')
ax[3].set_ylabel('P_bat')
