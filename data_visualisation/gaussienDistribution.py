#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 15:18:52 2020

@author: bahalla
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
mean = 10.75
variance = 21.64
sigma = math.sqrt(variance)
N=15
x=[3,3.5,4,4.5,5,5.5,6,6,7,7.5,8,8,9,9.5,10,10.5,10.75,11.5,
12,12.5,12.25,12.25,12,14.5,15,15.5,16,16.5,17,17.5,18,18.5]
y=stats.norm.pdf(x, mean, sigma)
plt.plot(x, y)
plt.show()
plt.savefig('normal1.png', dpi=72, bbox_inches='tight')