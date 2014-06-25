'''
Created on 14 May 2014

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





# class listBools(np.ndarray):
#     def __new__(cls, input_array):
#         obj = np.asarray(input_array, dtype = np.bool).view(cls)
#         return obj


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
    
    def addPrimitives(self, types, operators, 
                      terminals = None, ephemerals = None):
        
        primitiveList = self.generatePrimitives(types, operators)
        
        for item in primitiveList:
            self.addPrimitive(*item)
            self.primitivesList.append(item)
        
        self.addTerminalsEphemerals(terminals, ephemerals)
            
    def addTerminalsEphemerals(self, terminals = None, ephemerals = None):
        if terminals:
            self.addTerminals(terminals)
        if ephemerals:
            self.addEphemeral(ephemerals)
                
    def generatePrimitives(self, types, operators):
        """
        This function generates the set of possible 
        input and output types for a set of operators
        """
        primitiveList = []
        for operator in operators:
            inTypes =  operator[1]
            retTypes = operator[2]
            #create set of input types
            inTypeSet = set()
            inTypeSet.add(tuple(inTypes))
            newInTypeSet = set()
            
            #Keep looping while we add different input types
            #This generates the set of possible input types
            while newInTypeSet != inTypeSet and not inTypeSet.issubset(set()):
                newInTypeSet = copy.copy(inTypeSet)
                for InType in inTypes:                
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
                if set(types['list']) & set(inputType):         
                    for elem in types[retTypes]:
                        if elem in types['list']:
                            primitiveList.append([operator[0], inputType, elem])
                else:
                    if retTypes not in types['list']:
                        primitiveList.append([operator[0], inputType, retTypes])

        return primitiveList        
        
                
    def addSpecificPrimitives(self, operators):
        """
        This method adds only the operators specified
        It doesn't generate a set of all possible input types
        """
        for op in operators:
            self.addPrimitive(*op)
            self.primitivesList.append(op)
    
    def addTerminals(self, terminals):
        for item in terminals:
            self.addTerminal(*item)
            
    def addEphemeral(self, ephemerals):
        for item in ephemerals: 
            self.addEphemeralConstant(*item)
                
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
            if set(types['list']) & set(inputType):         
                for elem in types[retTypes]:
                    if elem in types['list']:
                        primitiveList.append([operator[0], inputType, elem])
                        print inputType, elem
            else:
                if retTypes not in types['list']:
                    primitiveList.append([operator[0], inputType, retTypes])
                    
    return primitiveList

        
def safeDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 0
    
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
        return x * 0 * y
    
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
    
def momentFloat(a, m = 1):
    return moment(a, m)

def momentVector(a, m = 1):
    return vector(moment(a, m, axis = 0))

def momentPosition(a, m = 1):
    return position(moment(a, m, axis = 0))

def float10():
    return random.random()*10

def int10():
    return random.randint(0,10)

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
    

if __name__ == '__main__':
    inputTypes = [float, particleState, particleState,
                  particlePosition, particleVector, numberField]
    
    referenceTypes = {
                      'number': float,
                      'mass': particleState,
                      'radius': particleState,
                      'position': particlePosition,
                      'momentum': particleVector,
                      'box': numberField,
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
                          #[np.dot, [vector, vector, ], float],
                          #[np.dot, [vector, position, ], float],
                          #[np.dot, [particleVector, position, ], float],
                          #[np.dot, [vector, particlePosition, ], float],
                          #[np.dot, [particleVector, particlePosition, ], float],
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
    
    ephemerals = [["rand10", lambda: random.random() * 10, float], ]
    ephemerals = [["int10", lambda: random.random() * 10, int], ]
    
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
    
    with open('pset.pkl' , 'wb') as f:
        pickle.dump(psetDict, f)

        
# types = {
#           bool: [bool,], # listBools],
#           float: [float, particleState],
#           'list': [particlePosition, particleState, particleVector], # listBools]
#           'field': [numberField,]
#           }
#  
# operators = [
#              [operator.mul, (float, float), float],
#              [operator.add, (float, float), float],
#              [operator.sub, [float, float], float],
#              [safeDiv, [float, float], float],
#              [operator.neg, [float], float],
#              #[safePow, [float, float], float],
#              [np.sqrt, [float,], float],
#              ]
#  
# specific_operators = [
#                       #[operator.lt, [float, float], bool],
#                       #[operator.gt, [float, float], bool],
#                       [sum, [particleState,], float],
#                       #[if_then_else, [bool, float, float], float],
#                       #[doubleValue, [float, float,], twinFloat]
#                       ]
# 
# terminals = [
#              [True, bool],
#              [False, bool],
#              #[twinFloat(0,0), twinFloat],
#              ]
#  
# ephemerals = [
#               ["rand10", lambda: random.random() * 10, float],
#               #["randTwin", lambda: twinFloat(random.random(),random.random(),), twinFloat]
#               ]
# 
# if __name__ == '__main__':
#     
#     ## This generates the set of all possible input types.
#     for operator in operators:
#         inTypes =  operator[1]
#         retTypes = operator[2]
#         #create set of input types
#         inTypeSet = set()
#         inTypeSet.add(tuple(inTypes))
#         newInTypeSet = set()
#         while newInTypeSet != inTypeSet:
#             newInTypeSet = copy.copy(inTypeSet)
#             for InType in inTypes:
#             #Keep looping while we add different input types
#                 for element in newInTypeSet:
#                     for j, typeElement in enumerate(element):
#                         if InType == typeElement:
#                             for type in types[InType]:
#                                 elem = list(copy.copy(element))
#                                 elem[j] = type
#                                 inTypeSet.add(tuple(elem))
#         for inputType in inTypeSet:
#             print [operator[0], inputType, retTypes]
#             
#     generatePrimitives(types, operators)        
# 
#     pset = Primitives("MAIN", (particleState,), float, 'ARG') 
                
       


            
   