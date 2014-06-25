'''
Created on 25 Jun 2014

@author: mg542
'''

import numpy as np

import scoop.futures as futures

x = np.linspace(0,50,10000)

def evalFunc(func):
    return func(x)

if __name__ == '__main__':
    
    atlas = [np.sin, np.cos, np.log]
    
    output = list(futures.map(evalFunc, atlas))
    
    print output[0]