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


def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Simple oscillating masses example with two masses and two inputs.
    # States are the position and velocitiy of the two masses.

    # States struct (optimization variables):
    T_r   = model.set_variable(var_type = '_x', var_name = 'T_r')
    T_w   = model.set_variable(var_type = '_x', var_name = 'T_w')
    T_e   = model.set_variable(var_type = '_x', var_name = 'T_e')
    E_bat = model.set_variable(var_type = '_x', var_name = 'E_bat')
    _x = vertcat(T_r, T_w, T_e, E_bat)

    # Input struct (optimization variables):
    P_heat = model.set_variable(var_type = '_u', var_name = 'P_heat')
    P_cool = model.set_variable(var_type = '_u', var_name = 'P_cool')
    P_bat  = model.set_variable(var_type = '_u', var_name = 'P_bat')
    P_hvac = model.set_expression('P_hvac', P_heat - P_cool)
    _u = vertcat(P_hvac, P_bat)

    # Set time-varying parameters
    d_T   = model.set_variable(var_type = '_tvp', var_name = 'd_T')
    d_sr  = model.set_variable(var_type = '_tvp', var_name = 'd_sr')
    _d = vertcat(d_T, d_sr)

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    P_PV = model.set_expression('P_PV', 0.5 * d_sr)
    P_grid = model.set_expression('P_grid', P_PV - P_heat - P_cool - P_bat)


    A = np.array([[ 0.8511,  0.0541,  0.0707,  0.000],
                  [ 0.1293,  0.8635,  0.0055,  0.000],
                  [ 0.0989,  0.0032,  0.7541,  0.000],
                  [ 0.0000,  0.0000,  0.0000,  1.000]])

    B = np.array([[ 0.0035,  0.0000],
                  [ 0.0003,  0.0000],
                  [ 0.0002,  0.0000],
                  [ 0.0000,  1.0000]])

    E = np.array([[ 22.217,  1.7912],
                  [ 1.5376,  0.6944],
                  [ 103.18,  0.1032],
                  [ 0.0000,  0.0000]])
    E = 1e-3 * E


    x_next = A @ _x + B @ _u + E @ _d
    for i in range(_x.shape[0]):
        model.set_rhs(_x[i].name(), x_next[i, 0])

    model.setup()

    return model
