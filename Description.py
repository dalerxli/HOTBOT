'''
Created on 20 May 2014

@author: mg542
'''

import numpy as np
import pickle
import exceptions

dimensions = 3

class position(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype = np.float).view(cls)
        return obj

class vector(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype = np.float).view(cls)
        return obj

class listBools(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype = np.bool).view(cls)
        return obj

class particlePosition(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype = np.float).view(cls)
        return obj

class particleVector(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype = np.float).view(cls)
        return obj
    
class particleState(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype = np.float).view(cls)
        return obj
    
class numberField(np.ndarray):
    
    def __new__(cls, input_array, XYZ):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array, dtype = np.float).view(cls)
        # add the new attribute to the created instance
        obj.XYZ = XYZ

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self,obj):
        # reset the attribute from passed original object
        self.XYZ = getattr(obj, 'XYZ', None)
        # We do not need to return anything

class Description(np.ndarray):
    '''
    This the class in which a propagation of the reference is stored
    '''

    names = ['time', 'state']
    formats = ['float64', 'object']
    stateNames = ['number', 'mass', 'radius',
                  'position', 'momentum', 'Lx', 'Ly', 'Lz' ]# 'box', 'XYZ']
    stateFormats = ['u1', 'f4', 'f4', 
                    'f4', 'f4', 'f4', 'f4', 'f4']
    
    def __new__(cls, reference = None, N=None, L = [], ref=None):
        if N is None and reference is not None:
            length = len(reference['time'])
        elif N is not None:
            length = 1      
        elif ref is not None:
            length = len(ref)
        dtype = dict(names=cls.names, formats=cls.formats)
        obj = np.empty(length, dtype = dtype)
        obj = np.asarray(obj).view(cls)
        return obj.view(cls)
    
    def __init__(self, reference = None, N=None, L = [], ref=None):
        if N is None and reference is not None:
            self.loadReference(reference, L)
        elif N is not None:
            self.createEmpty(N)
        elif ref is not None:
            self['time'][:] = ref['time'][:]
            self['state'][:] = ref['state'][:]
        else:
            raise 'not enough inputs'
            
    def createEmpty(self, N):
        time = 0
        shapes = [1, (N,1), N, (N, 3), (N, 3), 1, 1, 1]
        formats = zip(self.stateFormats, shapes)
        statedtype = dict(names=self.stateNames, formats=formats)
        state = np.array(np.empty((1,), dtype=statedtype)[0], dtype=statedtype)
        self[0] = (time, state)
        pass
    
    def loadReference(self, reference, L = []):
        tlist = reference['time']
        for i, time in enumerate(tlist):
            state = map(lambda x: reference['state'][i][x], reference['state'][i].dtype.names)
            state = state + list(L)
            n = state[0].item()
            shapes = [1, (n,1), n, (n, 3), (n, 3), 1, 1, 1]# boxShape, boxShape]
            formats = zip(self.stateFormats, shapes)
            statedtype = dict(names=self.stateNames, formats=formats)
            state = np.array(tuple(state), dtype=statedtype)
            self[i] = (time, state)
        return self
    
    def evalCharacteristic(self, mapping):
        tlist = self['time']
        names = ['time', 'characteristic']
        formats = ['float64', 'float64']
        dtype = dict(names=names, formats=formats)
        
        def extractArgs(state):
            names = state.dtype.names
            arg = map(lambda x: state[x], names)
            return arg

        characlist = map(lambda state: mapping(*extractArgs(state)), self['state'])
        
        charac = np.array(zip(tlist, characlist), dtype=dtype)
        return charac

class Description2:
    '''
    classdocs
    ''' 
    types = {
              'number': float,
              'mass': particleState,
              'radius': particleState,
              'position': particlePosition,
              'momentum': particleVector,
              'box': numberField,
              }
    
    def __new__(cls):
        cls.nlist = []
        cls.mlist = []
        cls.rlist = []
        cls.poslist = []
        cls.plist = []
        cls.tlist = []
        cls.box   = []
        
        return cls

    def __init__(self, listMass, listRadius, listPositions, 
                 listMomementa, fieldBox, XYZ, time = None):
        '''
        Constructor
        '''
        self.nlist = [float(len(listMass)),]
        self.mlist = [particleState(listMass),]
        self.rlist = [particleState(listRadius),]
        self.poslist = [particlePosition(listMomementa),]
        self.plist = [particleVector(listPositions),]
        self.box   = [numberField(fieldBox, XYZ),]
        if not time:
            self.tlist = [0.,]
        else:
            self.tlist = [float(time),]
            
        self.typesDict = {
                          'number': self.nlist,
                          'mass': self.mlist,
                          'radius': self.rlist,
                          'position': self.poslist,
                          'momentum': self.plist,
                          'box': self.box,
                          }
        
        self.typesList = (
                          self.nlist,
                          self.mlist,
                          self.rlist,
                          self.poslist,
                          self.plist,
                          self.box,
                          )    
        
    def addPropagation(self, listMass, listRadius,
                       listPositions, listMomementa,
                       fieldBox, XYZ, time):
        
        tindex = np.array(self.tlist).searchsorted(time)
        
        self.nlist.insert(tindex, float(len(listMass)))
        self.mlist.insert( tindex, particleState(listMass))
        self.rlist.insert( tindex, particleState(listRadius))
        self.plist.insert( tindex, particlePosition(listMomementa))
        self.poslist.insert( tindex, particleVector(listPositions))    
        self.box.insert(   tindex, numberField(fieldBox, XYZ))
        self.tlist.insert( tindex, float(time))
        
    def evalCharacteristics(self, mapping):
        '''
        This calculates the characteristic for each time step
        '''
        characteristic = map(mapping, *self.typesList)
        return (characteristic, self.tlist)
    
    def initialState(self):
        initial = (
                   self.mlist[0], self.rlist[0], self.poslist[0],
                   self.plist[0], self.box[0], self.box[0].XYZ, 
                   self.tlist[0]
                   )
        return initial
    
    def pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self,f)
           
