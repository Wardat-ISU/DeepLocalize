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
for i in range(10000):
     
    number =np.random.randint(low=1, high=10000)
    
    print("{0}\t{1}".format(number,np.sum((digitize(number)))))