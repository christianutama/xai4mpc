import numpy as np
from mpc_blackbox import get_mpc_action
import scipy.special
import math
from matplotlib import pyplot as plt
from itertools import permutations, product
from scipy.optimize import minimize
import tensorflow as tf
import multiprocessing
from functools import partial
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    X = np.load('data/input_complete.npy')
    y = np.load('data/output_complete.npy')

    n_features = X.shape[1]

    # Create coalitions
    zeros = np.zeros((n_features, 54))
    for index, array in enumerate(zeros):
        array[index] = 1

    ones = np.ones((n_features, 54))
    for index, array in enumerate(ones):
        array[index] = 0

    coalitions = np.vstack([zeros, ones])

    # Calculate coalition weights
    def shap_kernel(coalition):
        M = len(coalition)
        z_ = np.sum(coalition)
        return (M - 1) / (scipy.special.comb(M, z_) * z_ * (M - z_))

    coalition_weights = np.array([shap_kernel(i) for i in coalitions])

    # Map coalitions to the original feature space and calculate average predictions
    np.random.seed(2034)
    model = tf.keras.models.load_model('models/nn_controller_complete.h5')
    shap_values_0 = []
    shap_values_1 = []
    theta_0 = np.mean(model.predict(X), axis=0)[0]
    theta_1 = np.mean(model.predict(X), axis=0)[0]


    def fun(a, theta_0, predictions):
        return np.sum([np.square(np.sum(np.multiply(i, a)) + theta_0 - k) * j for i, j, k in
                       zip(coalitions, coalition_weights, predictions)])  # +0.001*np.sum(np.abs(a))


    for index, x_init in enumerate(X[0:30]):
        new_X = x_init * coalitions
        n_samples = 100
        y_predict = model.predict(x_init.reshape(1, -1))
        y_0 = y_predict[0][0]
        y_1 = y_predict[0][1]
        average_predictions = []
        for x in new_X:
            mask = np.where(x == 0)[0]
            x_mask = np.delete(X, index, 0)
            x_mask = x_mask[:, mask]
            x_copy = np.repeat(x.reshape(1, -1), n_samples, axis=0)
            sub_values = x_mask[np.random.randint(0, x_mask.shape[0], n_samples)].flatten()
            x_copy[x_copy == 0] = sub_values
            # for MPC:
            x_copy = [np.array(i) for i in x_copy]  # convert to list of arrays for parallel calculation
            pool = multiprocessing.Pool(processes=4)
            preds = pool.map(get_mpc_action, x_copy)
            # for NN:
            # preds = model.predict(x_copy)
            average_predictions.append(np.mean(preds, axis=0))
        predictions = np.array(average_predictions)

        def con_0(a):
            return theta_0 + np.sum(a) - y_0

        def con_1(a):
            return theta_1 + np.sum(a) - y_1

        cons = {'type': 'eq', 'fun': con_0}
        res = minimize(fun, np.ones(54), args=(theta_0, predictions[:, 0]))
        shap_values_0.append(res.x)

        cons = {'type': 'eq', 'fun': con_1}
        res = minimize(fun, np.ones(54), args=(theta_1, predictions[:, 1]))
        shap_values_1.append(res.x)

    shap_values_0 = np.vstack(shap_values_0)
    shap_values_1 = np.vstack(shap_values_1)

    np.save('shap_values_0_mpc_50', shap_values_0)
    np.save('shap_values_1_mpc_50', shap_values_1)
