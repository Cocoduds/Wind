
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:00:08 2022

@author: paoloforloni and coltondudley
"""


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from numpy import random
import pandas as pd
from scipy.signal import savgol_filter

plt.close('all')

Datapoints = 6
#%%
#Loading data and inserting numpy nans
data=np.loadtxt("Calibration1Data.csv", delimiter = ',', encoding='utf-8-sig')
for i in range(len(data[0,:])):
    for j in range(len(data[:,i])):
        if data[j][i] == -1:
            data[j][i] = np.nan

temps=np.loadtxt("Calibration1Temps.csv", delimiter = ',', encoding='utf-8-sig')
for i in range(len(data[0,:])):
    for j in range(len(temps[:,i])):
        if temps[j][i] == -1:
            temps[j][i] = np.nan

time = np.linspace(0,len(data[:,0]), num=len(data[:,0]))


#%% Plotting data and average wind speed for heights
averages=[]
stds=[]           
heights = np.array([154,590,150,789,261,348])
for i in range(0,Datapoints):
    averages.append(np.nanmean(data[:,i]))
    stds.append(np.nanstd(data[:,i]))
    plt.figure(1)
    plt.plot(time, data[:,i])
plt.figure(2)
plt.errorbar(heights,averages,yerr=stds, ls='none', fmt='.')

average = np.nanmean(averages)
for i in range(0, Datapoints):
    changeFactor = average / averages[i]
    print(heights[i] , 'calibration = ', changeFactor)
    for j in range(len(data[:,i])):
        data[j][i] = data[j][i] * changeFactor
        
#%%CUT DAata
#First we split the data, we try and splt into 10 seconds intervals
#cut_winds=[]
#for i in range(0,Datapoints):
#    n=len(data[:,i])*0.8*0.1
#    newarr=np.array_split(data[:,i], n)
#    cut_winds.append(newarr)
#   
##We now have a bunch of cut winds, we can individually average them
#averages_split=[]
#for j in range (0,Datapoints):
#    average_list=[]
#    for i in range(0,int(n)):
#        avg=np.nanmean(cut_winds[j][i])
#        average_list.append(avg)
#    averages_split.append(np.average(average_list))
#plt.errorbar(heights,averages_split,yerr=stds, ls='none', fmt='.')


#%% interpolated wind spectrum
interpolated_data=[]
for i in range(0,Datapoints):
    data[0,i] = data [1,i]
    s = pd.Series(data[:,i])
    s = s.interpolate(method='linear', order=2)
    interpolated_data.append(s)

plt.figure(3)

interpolated_data=np.array(interpolated_data)

for i in range(0,Datapoints):
    plt.plot(time, interpolated_data[i])
