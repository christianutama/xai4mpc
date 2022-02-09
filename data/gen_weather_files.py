import numpy as np
import pandas as pd
import pickle
import pdb
import matplotlib.pyplot as plt
plt.ion()

""" Auxiliary function """
def extract_data(year, data):
    lb_idx = data[:,0] > year * 1e6
    ub_idx = data[:,0] < (year + 1) * 1e6
    idx = lb_idx & ub_idx
    exp_data = data[idx,1:]

    return exp_data


""" Load data for temperatures and solar radiation """
# temperature
temp_dic = pd.read_csv('./stundenwerte_TU_03098_19940101_20201231_hist/values.csv', delimiter = ';')
temp = temp_dic[['MESS_DATUM', 'TT_TU']].to_numpy()


# solar radiation
sr_dic = pd.read_csv('./stundenwerte_ST_03098_row/values.csv', delimiter = ';')

# remove exact minute of to measurement to enable comparing
sr_dic['MESS_DATUM'] = sr_dic['MESS_DATUM'].map(lambda x: int(str(x)[:-3]))
sr_dic = sr_dic[['MESS_DATUM', 'FG_LBERG', 'SD_LBERG']]


""" Extract data from the years given in the list and save it"""
# choose a year
year = 2009

# extract temperatures
temp_dic_mean = temp_dic[['MESS_DATUM', 'TT_TU']].where(temp_dic['MESS_DATUM'] > year * 1e6).where(temp_dic['MESS_DATUM'] < (year + 1) * 1e6)
temp_exp = temp_dic_mean.dropna().to_numpy()


# extract solar radation
sr_dic_mean = sr_dic[sr_dic['MESS_DATUM'] > year * 1e6]
sr_dic_mean = sr_dic_mean[sr_dic_mean['MESS_DATUM'] < (year + 1) * 1e6]
sr_dic_mean.loc[sr_dic_mean['FG_LBERG'] == -999.0] = np.nan
sr_dic_mean = sr_dic_mean.interpolate(limit=4)
sr_dic_exp = sr_dic_mean.to_numpy()

# add the types of radiation and scale to W/m^2
sr_exp = np.sum(sr_dic_exp[:,1:],axis=1) * 10000 / 3600

fig, ax = plt.subplots()
ax.plot(temp_exp)
fig, ax = plt.subplots()
ax.plot(sr_exp)

print(temp_exp.shape)

exp_dic = {'date': temp_exp[:,0].reshape(-1,1), 'T': temp_exp[:,1].reshape(-1,1), 'sr': sr_exp[:].reshape(-1,1)}
with open('./exttemp_and_solrad_' + str(year) + '.pkl', 'wb') as f:
    pickle.dump(exp_dic, f)
