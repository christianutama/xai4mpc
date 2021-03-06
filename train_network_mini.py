import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow import keras
import pdb

""" Params """
n_hidden_layers = 6
w_hidden_layers = 80
epochs = 2000
batch_size = 512


""" Load data """
with open(r'./data/training_data.pkl', 'rb') as f:
    data = pickle.load(f)
X_raw = data['X']
U_raw = data['U']
P_raw = data['P']
H = data['H']

# Summarized P_hvac = P_heat (P[:,0]) - P_cool (P[:,1])
P_hvac = [np.reshape(u_raw[0, 0] - u_raw[1, 0], (1, 1)) for u_raw in U_raw]
P_bat =  [np.reshape(u_raw[-1, 0], (1, 1)) for u_raw in U_raw]
U = [np.hstack([p_hvac, p_bat]) for (p_hvac, p_bat) in zip(P_hvac, P_bat)]


""" Scaling """
x_lb = np.array([[20.0, 15.0,  0.0,     0.0]])
x_ub = np.array([[23.0, 25.0, 50.0, 20000.0]])
# X_s = (X_raw - x_lb) / (x_ub - x_lb)
X_s = [(x.T - x_lb)/(x_ub - x_lb) for x in X_raw]
# X_s = [np.squeeze(x_s) for x_s in X_s]

u_lb = np.array([[-1000, -1000]])
u_ub = np.array([[ 1000,  1000]])
U_s = [(u - u_lb)/(u_ub - u_lb) for u in U]
# U_s = [np.squeeze(u_s) for u_s in U_s]

p_lb = np.array([[-10.0,    0.0]])
p_ub = np.array([[ 30.0, 1200.0]])
P_s = [(p - p_lb)/(p_ub - p_lb) for p in P_raw]
T_s = [np.reshape(p[:, 0], (1, -1)) for p in P_s]
# T_s = [np.squeeze(t_s) for t_s in T_s]
SR_s = [np.reshape(p[:,1], (1, -1)) for p in P_s]
# SR_s = [np.squeeze(sr_s) for sr_s in SR_s]

data_in = []
for x_s, t_s, sr_s in zip(X_s, T_s, SR_s):
    data_in.append(np.hstack([x_s, t_s, sr_s]).reshape(1, -1))

# only use a subset of the features
data_in = np.load('./data/input.npy')
U_s = np.load('./data/output.npy')
data_in = data_in[:, [0, 1, 2, 3, 4, 29]]

""" Build NN model """
# TODO: revert to original script
# inputs = keras.Input(shape=(data_in[0].shape[1],))
inputs = keras.Input(shape=(data_in.shape[1],))
x = keras.layers.Dense(w_hidden_layers,activation='relu')(inputs)
for _ in range(n_hidden_layers-1):
    x = keras.layers.Dense(w_hidden_layers,activation='relu')(x)
# TODO: revert to original script
# outputs = keras.layers.Dense(U_s[0].shape[1],activation='linear')(x)
outputs = keras.layers.Dense(U_s.shape[1],activation='linear')(x)

model = keras.Model([inputs], [outputs])
optimizer = keras.optimizers.Adam(learning_rate=5e-4, epsilon=1e-3)
model.compile(optimizer=optimizer, loss='mse')

early_stopping = keras.callbacks.EarlyStopping(patience=10)

# Train model
# TODO: revert to original script
# hist = model.fit(np.vstack(data_in), np.vstack(U_s), batch_size = batch_size, epochs= epochs, shuffle=True)
hist = model.fit(data_in, U_s, batch_size = batch_size, epochs= epochs, shuffle=True,
                 validation_split=0.2, callbacks=[early_stopping])

# save model
model.save('./models/nn_controller_mini.h5')

# save input-output
# np.save('./data/input_mini', np.vstack(data_in))
# np.save('./data/output_mini', np.vstack(U_s))
