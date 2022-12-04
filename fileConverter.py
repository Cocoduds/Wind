# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 19:43:02 2022

@author: dudle
"""

f = open('Data1.csv','r', encoding = 'utf-8')
contents = f.read()
file2 = contents.replace(',', '.')
file3 = file2.replace(';', ',')
file4 = file3.replace('---', '-1')

f  = open('Data1fixed.csv','w', encoding = 'utf-8')
f.write(file4)

