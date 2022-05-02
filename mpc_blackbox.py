from shap_mpc_ex import extract_T_sr, extract_state_vars, scale_control_actions
from template_model_4d import template_model
from template_mpc import template_mpc_shap
from template_simulator import template_simulator_shap
from casadi.tools import *
import do_mpc


def get_mpc_action(X):
    # X = X[0]
    T_SR = extract_T_sr(X)
    model = template_model()
    mpc = template_mpc_shap(model, T_SR)
    simulator = template_simulator_shap(model, T_SR)
    x0 = extract_state_vars(X)
    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()
    u0 = mpc.make_step(x0)
    if u0[0] > u0[1]:
        return scale_control_actions(np.array([u0[0][0], u0[-1][0]]))[0]
    else:
        return scale_control_actions(np.array([-u0[1][0], u0[-1][0]]))[0]
