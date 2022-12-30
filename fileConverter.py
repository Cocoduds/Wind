# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 19:43:02 2022

@author: dudle
"""
import numpy as np
import os

order = [1,2,3,4,5,6] #VERTICLE DETECTORS THEN HIGHEST TO LOWEST
permutation = [5,0,4,1,3,2] #ORDER = VERTICLE DETECTORS THEN HIGHEST TO LOWEST
file = 'Ice2'


Calibration = {348:1,
               789:2,
               150:3,
               261:4,
               154:5,
               590:6}




f = open('RawData/'+file+'.csv','r', encoding = 'utf-8')
contents = f.read()
file2 = contents.replace(',', '.')
file3 = file2.replace(';', ',')
file4 = file3.replace('---', '-1')

f  = open(file+'fixed.csv','w', encoding = 'utf-8')
f.write(file4)
f.close()



data=np.loadtxt(file+'fixed.csv', delimiter = ',', encoding='utf-8-sig', skiprows= 1, max_rows = 34, usecols = range(1,13))
data = np.delete(data,0,1)
data = np.delete(data,1,1)
data = np.delete(data,2,1)
data = np.delete(data,3,1)
data = np.delete(data,4,1)
data = np.delete(data,5,1)
idx = np.empty_like(permutation)
idx[permutation] = np.arange(len(permutation))
data[:, idx]  # return a rearranged copy
data[:] = data[:, idx]  # in-place modification of a

np.savetxt(file+'Data.csv', data, delimiter=',')


data=np.loadtxt(file+'fixed.csv', delimiter = ',', encoding='utf-8-sig', skiprows= 1, max_rows = 34, usecols = range(1,13))
data = np.delete(data,1,1)
data = np.delete(data,2,1)
data = np.delete(data,3,1)
data = np.delete(data,4,1)
data = np.delete(data,5,1)
data = np.delete(data,6,1)
idx = np.empty_like(permutation)
idx[permutation] = np.arange(len(permutation))
data[:, idx]  # return a rearranged copy
data[:] = data[:, idx]  # in-place modification of a

np.savetxt(file+'Temps.csv', data, delimiter=',')


os.remove(file+'fixed.csv')
