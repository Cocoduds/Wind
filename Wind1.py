# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


data=np.loadtxt("Data1fixed.csv", delimiter = ',', encoding='utf-8-sig', skiprows= 1, max_rows = 709, usecols = range(1,11))
temps=[]
winds=[]
time = np.linspace(0,707, num=708)
for i in range(0,10):
    if i%2==0:
        temps.append(data[:,i])
    else:
        winds.append(data[:,i])
        winds[i] = winds[i].replace('-1', np.nan)
        
        
for i in range(0,4):
    plt.plot(time, winds[i])


