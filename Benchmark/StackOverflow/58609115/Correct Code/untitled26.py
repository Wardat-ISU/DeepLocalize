#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:13:53 2020

@author: wardat
"""

import numpy as np
from math import log10

def digitize(x):
    n = int(log10(x))
    for i in range(n, -1, -1):
        factor = 10**i
        k = x // factor
        yield k
        x -= k * factor
        
# 
for i in range(3557):
     
    number =np.random.randint(low=10000000000000, high=10000000000000000)
    length = len(list(digitize(number)))
    
    if length == 1:
        print("{0}\t{1}".format(number,0))
    elif length == 2:
        print("{0}\t{1}".format(number,list(digitize(number))[0]+1))
    elif length == 3:
        print("{0}\t{1}".format(number,list(digitize(number))[0]+10))
    elif length == 4:
        print("{0}\t{1}".format(number,list(digitize(number))[0]+100))
    elif length == 5:
        print("{0}\t{1}".format(number,list(digitize(number))[0]+1000))
    elif length == 6:
        print("{0}\t{1}".format(number,list(digitize(number))[0]+10000))
    elif length == 7:
        print("{0}\t{1}".format(number,list(digitize(number))[0]+100000))
    elif length == 8:
        print("{0}\t{1}".format(number,list(digitize(number))[0]+1000000))
    elif length == 9:
        print("{0}\t{1}".format(number,list(digitize(number))[0]+10000000))
    elif length == 9:
        print("{0}\t{1}".format(number,list(digitize(number))[0]+10000000))