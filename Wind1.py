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

def data_fix(Data):
    fixed_data=[]
    for k in range(0,5):
        data=[]
        for j in range(0,len(Data[0])):
            data.append(Data[k][j][0])
        fixed_data.append(data)
    return fixed_data


Datapoints = 5
#%%
#Loading data and isnerting numpy nans
data=np.loadtxt("BigFieldData.csv", delimiter = ',', encoding='utf-8-sig')
for i in range(len(data[0,:])):
    for j in range(len(data[:,i])):
        if data[j][i] == -1:
            data[j][i] = np.nan

temps=np.loadtxt("BigFieldDataTemps.csv", delimiter = ',', encoding='utf-8-sig')
for i in range(len(data[0,:])):
    for j in range(len(temps[:,i])):
        if temps[j][i] == -1:
            temps[j][i] = np.nan

time = np.linspace(0,len(data[:,0]), num=len(data[:,0]))

#%%CALIBRATION


#%% Plotting data and average wind speed for heights
averages=[]
stds=[]           
heights = np.array([1,2,3,4,5])*0.45
for i in range(0,Datapoints):
    averages.append(np.nanmean(data[:,i]))
    stds.append(np.nanstd(data[:,i]))
    plt.figure(1)
    plt.plot(time, data[:,i])
plt.figure(2)
plt.errorbar(heights,averages,yerr=stds, ls='none', fmt='.')


#%%
#First we split the data, we try and splt into 10 seconds intervals
cut_winds=[]
for i in range(0,Datapoints):
    n=len(data[:,i])*0.8*0.1
    newarr=np.array_split(data[:,i], n)
    cut_winds.append(newarr)
   
#We now have a bunch of cut winds, we can individually average them
averages_split=[]
for j in range (0,Datapoints):
    average_list=[]
    for i in range(0,int(n)):
        avg=np.nanmean(cut_winds[j][i])
        average_list.append(avg)
    averages_split.append(np.average(average_list))
plt.errorbar(heights,averages_split,yerr=stds, ls='none', fmt='.')

#%% AUTOCORRELATIONS
#taomax = int(n/2)
#R_list=[]
#for i in range(0,Datapoints):
#    x=averages_split[i]
#    var=np.nanmean((data[:,i]-x)**2)
#    R=[]
#    for t in range(0,taomax):
#        tof=[]
#        for n in range(0,len(data[:,i])):
#            if n+int(t/0.8) < len(data[:,i]):
#                tof.append((data[n,i]-x)*(data[n+int(t/0.8),i]-x))
#            else:
#                pass
#        avg_tof = np.nanmean(tof)
#        R.append(avg_tof/var)
#    R_list.append(R)   
# 
#plt.figure(4)
#for b in range(0,Datapoints):
#    plt.plot(np.linspace(0,taomax,27), R_list[b],label='position '+str(heights[b])+'m')
#plt.xlabel(r"$\tau$")
#plt.ylabel(r"$R(\tau)$")
#plt.legend()
#
##random numbers
#rand=random.rand(len(data[:,0]),1)
#
#x=np.mean(rand)
#var=np.nanmean((rand-x)**2)
#R_rand=[]
#for t in range(0,taomax):
#    tof=[]
#    for n in range(0,len(rand)):
#        if n+int(t/0.8) < len(rand):
#            tof.append((rand[n]-x)*(rand[n+int(t/0.8)]-x))
#        else:
#            pass
#    avg_tof = np.nanmean(tof)
#    R_rand.append(avg_tof/var)
#plt.plot(np.linspace(0,taomax,27), R_rand,label='Random numbers',linestyle="dotted",color='r', linewidth=2)

#%% PANDAS autocorelate
plt.figure(5)
autocorrsList = []
interpolated_data=[]
for i in range(0,Datapoints):
    autocorrs = []
    data[0,i] = data [1,i]
    s = pd.Series(data[:,i])
    s = s.interpolate(method='linear', order=2)
    interpolated_data.append(s)
    x = pd.plotting.autocorrelation_plot(s)
    x.plot()
    for j in range(0,len(data[:,i])):
        autocorrs.append(s.autocorr(j))
    autocorrsList.append(autocorrs)
plt.show()

#%% interpolated wind spectrum
plt.figure(6)

interpolated_data=np.array(interpolated_data)

for i in range(0,Datapoints):
    plt.plot(time, interpolated_data[i])
    
#%% FFT of wind spectrum
plt.figure(7)
for i in range (0,Datapoints):  
    tpCount     = len(interpolated_data[i]) 
    values      = np.arange(int(tpCount)) 
    timePeriod  = tpCount/1.25  
    frequencies = values/timePeriod 
    x=np.fft.fft(interpolated_data[i])
    plt.plot((frequencies*heights[i]/average_list[i]), average_list[i]* savgol_filter(abs(x),41,2)/(heights[i]*np.var(interpolated_data[i])))
    
plt.yscale('log')
plt.xscale('log')
plt.ylim(0,300)
plt.show()

#%% Richardson

averagetemps = []
for i in range(0,Datapoints):
    averagetemps.append(np.nanmean(temps[:,i]))
   
plt.figure(8)
plt.plot(heights,averagetemps)
plt.show()

fit_1,cov_1 = np.polyfit(heights,averages,1,cov=True)
p1=np.poly1d(fit_1)
print(p1)

fit_2,cov_2 = np.polyfit(heights,averagetemps,1,cov=True)
p2=np.poly1d(fit_2)
print(p2)

y= (9.81/np.mean(averagetemps))*(p2[1])
u=(p1[1]**2)
print(y/u)

#%% finding eddie size
def area_t(a,b,dx):
    return ((a+b)/2)*dx
   
dx=np.diff(time[:])
def integrate(y1,dx,tao):
    integral = 0
    for i in range(tao):
        integral = integral + area_t(y1[i+1], y1[i],dx[i])
    return(integral)

for j in range(0,Datapoints):
#    integal = 0
#    for i in range(0,len(data[i,:])):
#        integral = integal + data[i][j]
#    print(integral)
    print(integrate(autocorrsList[j],dx,40))
   
#%% Plotting fisrt log scale
fit_3,cov_3 = np.polyfit(np.log(heights),averages,1,cov=True)
p3=np.poly1d(fit_3)
print(p2)
x=np.linspace(-1,2,50)
plt.figure(9)
plt.scatter(np.log(heights),averages,label="Data")
plt.plot(x,p3(x),label="Linear Fit")
plt.title("Plot ln(z) against ū(z)")
print("The value of z_0 is =", np.exp(-p3[0]/p3[1]))


density = 1.225
dynamicV = 17.5e-6
kinematicV = 13.9e-6
lengthScale = 40
def UShear(density, dynamicV, gradient):
    shearStress = dynamicV * gradient
    return(np.sqrt(shearStress/density))
print('u* from shear stress = ',UShear(density,dynamicV, p1[1]))


def UDarcy(averages, kinematicV, lengthScale):
    averageWind = np.mean(averages)
    Re = averageWind*lengthScale/kinematicV #assuming open pipe? with disturbance on a length scale (40m)
    print(Re)
    fd = 64/Re # assuming laminar flow
    return averageWind*np.sqrt(fd/8)

print('u* from darcy friction factor = ',UDarcy(p1[1],kinematicV,lengthScale))

