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


def template_mpc(model, step_init = 0):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_robust': 0,
        'n_horizon': 24,
        't_step': 1.0,
        'store_full_solution':True,
    }

    mpc.set_param(**setup_mpc)

    mterm = - 1e-3 * model.x['E_bat']                # terminal cost
    # TODO: change objective
    lterm = - model.aux['P_grid'] + 1e-3 * model.u['P_heat'] + 1e-3 * model.u['P_cool'] # stage cost
    # lterm = model.u['P_bat'] + 1e-3 * model.u['P_heat'] + 1e-3 * model.u['P_cool']  # stage cost

    mpc.set_objective(lterm=lterm, mterm = mterm)
    # mpc.set_rterm(u=1e-4)

    mpc.bounds['lower','_x','T_r']    = 20.0
    mpc.bounds['upper','_x','T_r']    = 23.0

    # mpc.bounds['lower','_x','T_e']    =  0.0
    # mpc.bounds['upper','_x','T_e']    = 50.0
    #
    # mpc.bounds['lower','_x','T_w']    =  0.0
    # mpc.bounds['upper','_x','T_w']    = 50.0

    mpc.bounds['lower','_x','E_bat']  =     0.0
    mpc.bounds['upper','_x','E_bat']  = 20000.0

    mpc.bounds['lower','_u','P_heat'] =     0.0
    mpc.bounds['upper','_u','P_heat'] =  1000.0

    mpc.bounds['lower','_u','P_cool'] =     0.0
    mpc.bounds['upper','_u','P_cool'] =  1000.0

    mpc.bounds['lower','_u','P_bat']  = -1000.0
    mpc.bounds['upper','_u','P_bat']  =  1000.0

    # ensure energy balance |P_hvac| <= P_PV + P_grid - P_bat with P_PV = energy from solar panel

    P_PV   = model.aux['P_PV']
    P_heat = model.u['P_heat']
    P_cool = model.u['P_cool']
    P_grid = model.aux['P_grid']
    P_bat  = model.u['P_bat']
    mpc.set_nl_cons('energy_ub',   P_PV - P_heat - P_cool - P_grid - P_bat, ub = 0.0)
    mpc.set_nl_cons('energy_lb', - P_PV + P_heat + P_cool + P_grid + P_bat, ub = 0.0)
    mpc.set_nl_cons('P_grid_ub',   P_grid, ub =  2000.0)
    mpc.set_nl_cons('P_grid_lb', - P_grid, ub =  2000.0)


    with open(r'./data/exttemp_and_solrad_2009.pkl', 'rb') as f:
        data = pickle.load(f)
    T_data = data['T']
    sr_data = data['sr']
    tvp_temp_mpc = mpc.get_tvp_template()

    def tvp_fun(t_now):
        step = int(t_now) + step_init
        for i in range(len(tvp_temp_mpc['_tvp'])):
            tvp_temp_mpc['_tvp', i] = DM([T_data[step+i,0], sr_data[step+i,0]])

        return tvp_temp_mpc

    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()

    return mpc
