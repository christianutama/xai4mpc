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


""" Params """
n_trajectories = 6000
n_days = 1


"""
Set initial state
"""

x0_min = np.array([[20.5], [18.0], [18.0], [ 5000.0]])
x0_max = np.array([[22.5], [25.0], [25.0], [15000.0]])

model = template_model()

X0 = []
U0 = []
P0 = []
H0 = []

for i in range(n_trajectories):
    print(f'TRAJECTORY NO: {i}')

    """
    Get configured do-mpc modules:
    """
    # init_offset = np.random.randint(2000 - (n_days + 1) * 24 - 1)
    if i % 2 == 0:
        init_offset = np.random.randint(0, 576)
    else:
        init_offset = np.random.randint(5087, 5663)
    # init_offset = np.random.randint(5087, 5663)
    mpc = template_mpc(model, init_offset)
    simulator = template_simulator(model, init_offset)
    estimator = do_mpc.estimator.StateFeedback(model)

    # Set initial state
    x0 = np.random.uniform(x0_min,x0_max)
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()


    """
    Run MPC main loop:
    """

    for k in range(24):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)


        # Append data
        X0.append(np.reshape(x0, (-1, 1)))
        U0.append(np.reshape(u0, (-1, 1)))
        P0.append(np.hstack(mpc.opt_p_num['_tvp']).T)
        H0.append(k % 24)

        x0 = estimator.make_step(y_next)

    # plt.ion()
    # fig, ax = plt.subplots(3,1)
    # ax[0].plot(simulator.data['_x','T_r'], label = 'T_r')
    # ax[0].plot(simulator.data['_x','T_e'], label = 'T_e')
    # ax[0].plot(simulator.data['_x','T_w'], label = 'T_w')
    # ax[0].legend()
    # ax[1].plot(simulator.data['_aux','P_hvac'], label = 'P_hvac')
    # # ax[1].plot(simulator.data['_u','P_heat'], label = 'P_heat')
    # # ax[1].plot(simulator.data['_u','P_cool'], label = 'P_cool')
    # ax[1].plot(simulator.data['_u','P_bat'], label = 'P_bat')
    # ax[1].plot(simulator.data['_aux','P_grid'], label = 'P_grid')
    # ax[1].plot(simulator.data['_aux','P_PV'], label = 'P_PV')
    # ax[1].legend()
    # ax[2].plot(simulator.data['_x','E_bat'])
    # pdb.set_trace()



exp_dic = {'X': X0, 'U': U0, 'P': P0, 'H':H0}
with open('./data/training_data_complete.pkl', 'wb') as f:
    pickle.dump(exp_dic, f)
