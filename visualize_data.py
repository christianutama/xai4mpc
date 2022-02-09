import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.ion()


""" Load data """
with open(r'./data/exttemp_and_solrad_2008.pkl', 'rb') as f:
    data = pickle.load(f)
T_data = data['T']
sr_data = data['sr']


""" Plot data """
for _ in range(10):
    i = np.random.randint(T_data.shape[0] - 120)
    fig, ax = plt.subplots(2,1)
    ax[0].plot(T_data[i:i+120])
    ax[1].plot(sr_data[i:i+120])
