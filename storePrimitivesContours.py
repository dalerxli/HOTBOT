'''
Created on 6 Jun 2014

@author: mg542
'''

import copy
import operator
import random
import exceptions
import pickle

from deap import gp
import numpy as np
#from scipy.stats import moment

from Description import *
from Primitives import *


    


inputTypes = [float, CellTypes, CellStates, CellStates,
              CellPositions, CellVectors, Mesh]

inputNames = ['number', 'types', 'radii', 'lengths', 'positions', 'directions', 'mesh']
    
referenceTypes = {
                  'number': float, 
                  'cellType': CellTypes,
                  'length': CellStates,
                  'radius': CellStates,
                  'position': CellPositions,
                  'direction': CellVectors,
                  'mesh': Mesh
                  }
types = {
         bool: [bool, ListBools, ],
         int: [CellTypes, ],
         float: [float, CellStates, NumberField, Density],
         Vector: [Vector, Position, CellPositions, CellVectors, ],
         'list': [ListBools, CellStates, CellPositions, CellVectors, ],
         'listVector': [CellPositions, CellVectors, ],
         'field': [NumberField, Density],
         'contour': [Contour,],
         'contours': [Contours,],
         }

operators = [[np.multiply, (float, float,), float],
             [np.multiply, (Vector, Vector), Vector],
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
             [np.linalg.norm, [Vector, ], float],
             [np.cross, [Vector, Vector, ], Vector],
             #[identity, [Vector, ], Vector],
             ]

specific_operators = [
                      [np.dot, [Vector, Vector, ], float],
                      [np.dot, [Vector, Position, ], float],
                      [meshFind, [CellPositions, float], Mesh ],
                      #[np.dot, [CellVectors, Position, ], CellStates],
                      #[np.dot, [Vector, CellPositions, ], CellStates],
                      #[np.dot, [CellVectors, CellPositions, ], CellStates],
                      [vectorX, [Vector, ], float],
                      [vectorY, [Vector, ], float],
                      [vectorZ, [Vector, ], float],
                      [vectorX, [Position, ], float],
                      [vectorY, [Position, ], float],
                      [vectorZ, [Position, ], float],
                      [lVectorX, [CellVectors, ], CellStates],
                      [lVectorY, [CellVectors, ], CellStates],
                      [lVectorZ, [CellVectors, ], CellStates],
                      [lVectorX, [CellPositions, ], CellStates],
                      [lVectorY, [CellPositions, ], CellStates],
                      [lVectorZ, [CellPositions, ], CellStates],
                      [NumberField, [CellPositions, CellStates, Mesh], NumberField],
                      [NumberField, [CellPositions, CellStates, Mesh, CellTypes, CellType], NumberField],
                      [Density, [CellPositions, Mesh,], Density],
                      [Density, [CellPositions, Mesh, CellTypes, CellType], Density],
                      [densityMultiply, [Density, Density], Density],
                      [densityMultiply, [NumberField, NumberField], NumberField],
                      [densityMultiply, [Density, NumberField], Density],
                      [densityMultiply, [NumberField, Density], NumberField],
                      [Contours, [NumberField, float, Mesh], Contours],
                      [Contours, [Density, float, Mesh], Contours],
                      [largestContour, [Contours,], Contour],
                      [sumPerimeters, [Contours,], float],
                      [sumAreas, [Contours,], float],
                      [avgSolidity, [Contours,], float],
                      [perimeter, [Contour,], float],
                      [area, [Contour,], float],
                      [convexHull, [Contour,], Contour],
                      [centroidX, [Contour,], float],
                      [centroidY, [Contour,], float],
                      [solidity, [Contour,], float],
                      [orientation, [Contour,], float],
                      [majorAxis, [Contour,], float],
                      [minorAxis, [Contour,], float],
                      [nematicOrder, [CellPositions, CellVectors, Mesh,], NumberField],
                      [nematicOrder, [CellPositions, CellVectors, Mesh, CellTypes, CellType], NumberField],
                      [adfContour, [CellPositions, Mesh,], Contour],
                      [adfContour, [CellPositions, Mesh, CellTypes, CellType], Contour],
                      ]

terminals = [[True, bool],
             [False, bool],
             [CellType(0), CellType, 'CellType0'],
             [CellType(1), CellType, 'CellType1'], ]


ephemerals = [
              ["rand10", float10, float], 
              ["int10", int10, int], 
              #["cellType", CellType, CellType],
              ]

 
if __name__ == '__main__':   
    pset = Primitives("MAIN", inputTypes, float, 'ARG')
    pset.renameArgs(inputNames)
    pset.addPrimitives(types, operators, terminals, ephemerals)
    pset.addSpecificPrimitives(specific_operators)
    
    psetDict = {
                'inputTypes': inputTypes,
                'inputNames': inputNames,
                'types': types,
                'operators': operators,
                'terminals': terminals,
                'ephemerals': ephemerals,
                'specificOperators': specific_operators,
                }   
    
    with open('/home/mg542/Source/HOTBOT/pset.pkl' , 'wb') as f:
        pickle.dump(psetDict, f)
