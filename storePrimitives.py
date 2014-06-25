'''
Created on 6 Jun 2014

@author: mg542
'''

import copy
import operator
import random
import exceptions

from deap import gp
import numpy as np
from scipy.stats import moment

from Description import *
from Primitives import *

def createPset(**psetDict):
    inputTypes = psetDict['inputTypes']
    types = psetDict['types']
    operators = psetDict['operators']
    terminals = psetDict['terminals']
    ephemerals = psetDict['ephemerals']
    specific_operators = psetDict['specificOperators']
    
    pset = Primitives("MAIN", inputTypes, float, 'ARG')
    pset.addPrimitives(types, operators, terminals, ephemerals)
    pset.addSpecificPrimitives(specific_operators)
    
    return pset

def loadPset(filename):
    with open(filename, 'rb') as f:
        psetDict = pickle.load(f)
    return createPset(**psetDict)
    


inputTypes = [float, particleState, particleState,
              particlePosition, particleVector, float, float, float]

referenceTypes = {
                  'number': float,
                  'mass': particleState,
                  'radius': particleState,
                  'position': particlePosition,
                  'momentum': particleVector,
                  }
types = {
         bool: [bool, listBools, ],
         float: [float, particleState, ],
         vector: [vector, position, particlePosition, particleVector, ],
         'list': [listBools, particleState, particlePosition, particleVector, ],
         'listVector': [particlePosition, particleVector, ],
         'field': [numberField, ],
         }

operators = [
             [np.multiply, (float, float,), float],
             [np.multiply, (vector, vector), vector],
             [np.add, (float, float,), float],
             [np.subtract, [float, float, ], float],
             [safeDiv, [float, float, ], float],
             [np.negative, [float, ], float],
             [np.sqrt, [float, ], float],
             [safePower, [float, float, ], float],
             [np.exp, [float, ], float],
             [np.log, [float, ], float],
             [np.abs, [float, ], float],
             [np.greater, [float, float, ], bool],
             [np.less, [float, float, ], bool],
             [np.linalg.norm, [vector, ], float],
             [np.cross, [vector, vector, ], vector],
             #[np.dot, [vector, vector, ], float],
             [identity, [vector, ], vector],
             ]

specific_operators = [
                      [dot, [vector, vector, ], float],
                      [dot, [vector, position, ], float],
                      [ListDot, [particleVector, position, ], particleState],
                      [ListDot, [vector, particlePosition, ], particleState],
                      [ListDot, [particleVector, particlePosition, ], particleState],
                      [vectorX, [vector, ], float],
                      [vectorY, [vector, ], float],
                      [vectorZ, [vector, ], float],
                      [vectorX, [position, ], float],
                      [vectorY, [position, ], float],
                      [vectorZ, [position, ], float],
                      [lVectorX, [particleVector, ], particleState],
                      [lVectorY, [particleVector, ], particleState],
                      [lVectorZ, [particleVector, ], particleState],
                      [lVectorX, [particlePosition, ], particleState],
                      [lVectorY, [particlePosition, ], particleState],
                      [lVectorZ, [particlePosition, ], particleState],
                      [np.sum, [particleState, ], float],
                      [sumVector, [particleVector,], vector],
                      [sumVector, [particlePosition,], position],
                      [if_then_else, [bool, float, float,], float],
                      # [momentFloat, [particleState, int], float],
                      # [momentVector, [particleVector, int], vector],
                      # [momentPosition, [particlePosition, int], position],
                      ]

terminals = [[True, bool],
             [False, bool], ]

ephemerals = [["rand10", float10, float], 
              ["int10", int10, int], ]
 
if __name__ == '__main__':   
    pset = Primitives("MAIN", inputTypes, float, 'ARG')
    pset.addPrimitives(types, operators, terminals, ephemerals)
    
    pset.addSpecificPrimitives(specific_operators)
    
    psetDict = {
                'inputTypes': inputTypes,
                'types': types,
                'operators': operators,
                'terminals': terminals,
                'ephemerals': ephemerals,
                'specificOperators': specific_operators,
                }   
    
    with open('/home/mg542/Documents/Source/Gas/pset.pkl' , 'wb') as f:
        pickle.dump(psetDict, f)