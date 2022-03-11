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
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
import pickle


def template_simulator(model, step_init = 0):
    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = 1.0)

    # load data
    with open(r'./data/exttemp_and_solrad_2009.pkl', 'rb') as f:
        data = pickle.load(f)
    d_T_data = data['T']
    d_sr_data = data['sr']

    tvp_temp_sim = simulator.get_tvp_template()

    def tvp_fun(t_now):
        step = int(t_now) + step_init
        tvp_temp_sim['d_T'] = d_T_data[step,0]
        tvp_temp_sim['d_sr'] = d_sr_data[step,0]
        return tvp_temp_sim

    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    return simulator


def template_simulator_shap(model, T_sr_data):
    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = 1.0)

    # load data
    d_T_data = T_sr_data[:, 0:1]
    d_sr_data = T_sr_data[:, 1:]

    tvp_temp_sim = simulator.get_tvp_template()

    def tvp_fun(t_now):
        step = int(t_now)
        tvp_temp_sim['d_T'] = d_T_data[step,0]
        tvp_temp_sim['d_sr'] = d_sr_data[step,0]
        return tvp_temp_sim

    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    return simulator

