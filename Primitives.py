'''
Created on 14 May 2014

@author: mg542
'''

import pickle
import copy
import operator
import random
import exceptions

from deap import gp
import numpy as np
#from scipy.stats import moment

from Description import *


class Primitives(gp.PrimitiveSetTyped):
    '''
    This class is wrapper to the PrimitiveSetTyped class 
    to make it easier to add complex typed primitives
    '''
    primitivesList = []
    
    def __new__(cls, name, in_types, ret_type, prefix="ARG",
                types = None,
                operators = None,
                terminal = None,
                ephemerals = None
                ):
        pset = gp.PrimitiveSetTyped.__new__(cls, name, in_types, ret_type, prefix="ARG")
        return pset
    
    def renameArgs(self, inputNames):
        kwargs = dict((''.join(("ARG", str(i))), name) for i, name in enumerate(inputNames))
        self.renameArguments(**kwargs)
    
    def addPrimitives(self, types, operators, 
                      terminals = None, ephemerals = None):
        
        primitiveList = self.generatePrimitives(types, operators)
            
        for item in primitiveList:
            self.addPrimitive(*item, name=namePrimitive(item))
            self.primitivesList.append(item)
        
        self.addTerminalsEphemerals(terminals, ephemerals)
            
    def addTerminalsEphemerals(self, terminals = None, ephemerals = None):
        if terminals:
            self.addTerminals(terminals)
        if ephemerals:
            self.addEphemeral(ephemerals)
                
    def generatePrimitives(self, types, operators):  
        #This function generates the set of possible 
        #input and output types for a set of operators   
        primitiveList = []
        for operator in operators:
            inTypes =  operator[1]
            retTypes = operator[2]
            #create set of input types
            inTypeSet = set()
            inTypeSet.add(tuple(inTypes))
            newInTypeSet = set()
            while newInTypeSet != inTypeSet:
                newInTypeSet = copy.copy(inTypeSet)
                for InType in inTypes:
                #Keep looping while we add different input types
                    for element in newInTypeSet:
                        for j, typeElement in enumerate(element):
                            if InType == typeElement:
                                for type in types[InType]:
                                    elem = list(copy.copy(element))
                                    elem[j] = type
                                    inTypeSet.add(tuple(elem))       
            #This makes sure if one of the inputs is a list
            #then the output will be a list.                        
            for inputType in inTypeSet:
                #Checking any primitives using fields
                if set(types['field']) & set(inputType):
                    #Checks that there is only one field type present in inputs
                    if len([type for type in inputType if type in types['field']]) == 1:
                        #print len([type for type in inputType if type in types['field']])
                        #Check that no lists are present in input types
                        if not set(types['list']) & set(inputType):
                            #Adds appropriate primitive
                            for elem in types['field']:
                                if elem in inputType:
                                    primitiveList.append([operator[0], inputType, elem])
                                    #print inputType, elem
                #Checking any primitive using lists
                elif set(types['list']) & set(inputType):         
                    for elem in types[retTypes]:
                        if elem in types['list']:
                            if not set(types['field']) & set(inputType):
                                primitiveList.append([operator[0], inputType, elem])
                                #print inputType, elem
                else:
                    if retTypes not in types['list'] and retTypes not in types['field']:
                        primitiveList.append([operator[0], inputType, retTypes])                    
        return primitiveList        
                
    def addSpecificPrimitives(self, operators):
        """
        This method adds only the operators specified
        It doesn't generate a set of all possible input types
        """
        for op in operators:
            self.addPrimitive(*op, name=namePrimitive(op))
            self.primitivesList.append(op)
    
    def addTerminals(self, terminals):
        for item in terminals:
            self.addTerminal(*item)
            
    def addEphemeral(self, ephemerals):
        for item in ephemerals: 
            self.addEphemeralConstant(*item)
            
def namePrimitive(args):
    funcName = args[0].__name__
    argNames = '-'.join(map(lambda arg: arg.__name__, args[1]))
    name = '_'.join((funcName, argNames))
    return name
                
def generatePrimitives(types, operators):  
    #This function generates the set of possible 
    #input and output types for a set of operators   
    primitiveList = []
    for operator in operators:
        inTypes =  operator[1]
        retTypes = operator[2]
        #create set of input types
        inTypeSet = set()
        inTypeSet.add(tuple(inTypes))
        newInTypeSet = set()
        while newInTypeSet != inTypeSet:
            newInTypeSet = copy.copy(inTypeSet)
            for InType in inTypes:
            #Keep looping while we add different input types
                for element in newInTypeSet:
                    for j, typeElement in enumerate(element):
                        if InType == typeElement:
                            for type in types[InType]:
                                elem = list(copy.copy(element))
                                elem[j] = type
                                inTypeSet.add(tuple(elem))       
        #This makes sure if one of the inputs is a list
        #then the output will be a list.                        
        for inputType in inTypeSet:
            #Checking any primitives using fields
            if set(types['field']) & set(inputType):
                #Checks that there is only one field type present in inputs
                if len([type for type in inputType if type in types['field']]) == 1:
                    #print len([type for type in inputType if type in types['field']])
                    #Check that no lists are present in input types
                    if not set(types['list']) & set(inputType):
                        #Adds appropriate primitive
                        for elem in types['field']:
                            if elem in inputType:
                                primitiveList.append([operator[0], inputType, elem])
                                #print inputType, elem
            #Checking any primitive using lists
            elif set(types['list']) & set(inputType):         
                for elem in types[retTypes]:
                    if elem in types['list']:
                        if not set(types['field']) & set(inputType):
                            primitiveList.append([operator[0], inputType, elem])
                            #print inputType, elem
            else:
                if retTypes not in types['list'] and retTypes not in types['field']:
                    primitiveList.append([operator[0], inputType, retTypes])                    
    return primitiveList

        
def safeDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 0.
    
def safePower(x1, x2):
    return np.nan_to_num(np.power(x1, x2))

def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2
    
def safePow(x, y):
    try:
        return pow(x, y)
    except exceptions.ValueError:
        try:
            return pow(x * (x > 0), y)
        except exceptions.ZeroDivisionError:
            return x * 0 * y
    except exceptions.OverflowError:
        return float("inf")
    except exceptions.ZeroDivisionError:
        return x * 0. * y
    
def vectorX(vec):
    return vec[0]

def vectorY(vec):
    return vec[0]

def vectorZ(vec):
    return vec[0]

def lVectorX(lVec):
    return lVec[:,0]

def lVectorY(lVec):
    return lVec[:,1]

def lVectorZ(lVec):
    return lVec[:,2]

def identity(x):
    return x

def sumVector(x):
    return x.sum(axis=0)

def ListDot(a, b):
    ab = a*b
    return ab.sum(axis=1)

def dot(a, b):
    return sum(a*b)
    
# def momentFloat(a, m = 1):
#     return moment(a, m)
# 
# def momentVector(a, m = 1):
#     return Vector(moment(a, m, axis = 0))
# 
# def momentPosition(a, m = 1):
#     return Position(moment(a, m, axis = 0))

def float10():
    return random.random()*10

def int10():
    return random.randint(0,10)

def largestContour(contours):
    return contours.largestContour()

def sumPerimeters(contours):
    return contours.sumPerimeters()

def perimeter(contour):
    return contour.perimeter()

def area(contour):
    return contour.area()

def convexHull(contour):
    return contour.convexHull()

def densityMultiply(dens1, dens2):
    dens2 = reMesh(dens1, dens2)
    return dens1 * dens2

def densityAdd(dens1, dens2):
    dens2 = reMesh(dens1, dens2)
    return dens1 + dens2

def densityDivide(dens1, dens2):
    dens2 = reMesh(dens1, dens2)
    return dens1 / dens2

def centroidX(contour):
    pos = contour.centroid()
    return pos[0]

def centroidY(contour):
    pos = contour.centroid()
    return pos[1]

def solidity(contour):
    area = contour.area()
    hullArea = contour.convexHull().area()
    return float(area) / hullArea
    
def orientation(contour):
    (x,y),(MA,ma),angle = contour.fitEllipse()
    return angle

def majorAxis(contour):
    (x,y),(MA,ma),angle = contour.fitEllipse()
    return Ma

def minorAxis(contour):
    (x,y),(MA,ma),angle = contour.fitEllipse()
    return ma

def nematicOrder(cellPositions, cellDirection, mesh, types=None, cellType=None):
    p2 = 1.5 * np.cos(cellDirection[:,0])**2 - 0.5
    nOrder = NumberField(cellPositions, p2, mesh, types, cellType)
    return nOrder

def adfContour(cellPositions, mesh, types=None, cellType=None):
    density = Density(cellPositions, mesh, types, cellType)
    contours = Contours(density, 1, mesh)
    contour = contours.largestContour()
    return contour

def createPset(**psetDict):
    inputTypes = psetDict['inputTypes']
    inputNames = psetDict['inputNames']
    types = psetDict['types']
    operators = psetDict['operators']
    terminals = psetDict['terminals']
    ephemerals = psetDict['ephemerals']
    specific_operators = psetDict['specificOperators']
    
    pset = Primitives("MAIN", inputTypes, float, 'ARG')
    pset.renameArgs(inputNames)
    pset.addPrimitives(types, operators, terminals, ephemerals)
    pset.addSpecificPrimitives(specific_operators)
    
    return pset

def loadPset(filename):
    with open(filename, 'rb') as f:
        psetDict = pickle.load(f)
    return createPset(**psetDict)
    

if __name__ == '__main__':
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
                          [np.dot, [CellVectors, Position, ], CellStates],
                          [np.dot, [Vector, CellPositions, ], CellStates],
                          [np.dot, [CellVectors, CellPositions, ], CellStates],
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
                          [np.sum, [CellStates, ], float],
                          [sumVector, [CellVectors,], Vector],
                          [sumVector, [CellPositions,], Position],
                          [if_then_else, [bool, float, float,], float],
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
                          [perimeter, [Contour,], float],
                          [area, [Contour,], float],
                          [convexHull, [Contour,], Contour],
                          # [momentFloat, [CellStates, int], float],
                          # [momentVector, [CellVectors, int], vector],
                          # [momentPosition, [CellPositions, int], Position],
                          ]
    
    terminals = [[True, bool],
                 [False, bool], ]
    

    ephemerals = [["rand10", lambda: random.random() * 10, float], 
                  ["int10", lambda: random.random() * 10, int], 
                  ["cellType", lambda:CellType(), CellType], ]

    
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
    
    operators = [[np.multiply, (float, float,), float],]
    prim = generatePrimitives(types, operators)



            
   
