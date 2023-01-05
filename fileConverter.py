# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 19:43:02 2022

@author: dudle
"""
import numpy as np
import os

permutation = [4,0,2,3,1] #lowest to highest then verticle
file = 'day2-1'

Calibration = {
    154 : 0.8447024013209711,
    590 : 0.8789918044369084,
    150 : 1.0803072140599494,
    789 : 1.059367134308126,
    261 : 0.9092088339325909,
    348 : 1.410428641068372
    }



calibration = [Calibration[789],Calibration[348],Calibration[590],Calibration[154],1]



f = open('RawData/'+file+'.csv','r', encoding = 'utf-8')
contents = f.read()
file2 = contents.replace(',', '.')
file3 = file2.replace(';', ',')
file4 = file3.replace('---', '-1')

f  = open(file+'fixed.csv','w', encoding = 'utf-8')
f.write(file4)
f.close()



data=np.loadtxt(file+'fixed.csv', delimiter = ',', encoding='utf-8-sig', skiprows= 1, usecols = range(1,11))
data = np.delete(data,0,1)
data = np.delete(data,1,1)
data = np.delete(data,2,1)
data = np.delete(data,3,1)
data = np.delete(data,4,1)
data = data[:, permutation]
for i in range(len(data[1,:])):
    for j in range(len(data[:,i])):
        data[j,i] = data[j,i] * calibration[i] 
# idx = np.empty_like(permutation)
# idx[permutation] = np.arange(len(permutation))
# data[:, idx]  # return a rearranged copy
# data[:] = data[:, idx]  # in-place modification of a

np.savetxt(file+'Data.csv', data, delimiter=',')


data=np.loadtxt(file+'fixed.csv', delimiter = ',', encoding='utf-8-sig', skiprows= 1, usecols = range(1,11))
data = np.delete(data,1,1)
data = np.delete(data,2,1)
data = np.delete(data,3,1)
data = np.delete(data,4,1)
data = np.delete(data,5,1)
data = data[:, permutation]

np.savetxt(file+'Temps.csv', data, delimiter=',')


# os.remove(file+'fixed.csv')
