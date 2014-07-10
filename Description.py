'''
Created on 20 May 2014

@author: mg542
'''
import os
import numpy as np
#from scipy.integrate import dblquad
import cPickle
import cv2

from random import randint

from CellModeller.CellState import CellState


directory = '/scratch/mg542/Data/CellModeller'
pickleFile = '/scratch/mg542/Data/CellModeller/1/20140704-204113/step-00550.pickle'

class Position(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype = np.float).view(cls)
        return obj

class Vector(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype = np.float).view(cls)
        return obj

class ListBools(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype = np.bool).view(cls)
        return obj

class CellPositions(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype = np.float).view(cls)
        return obj

class CellVectors(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype = np.float).view(cls)
        return obj
    
class CellStates(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype = np.float).view(cls)
        return obj
    
class CellTypes(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype = np.int).view(cls)
        return obj
    
class CellType(int):
    def __new__(self, type=None):
        if type is not None:
            obj = int(type)
        else:
            obj = randint(0,1)
        return obj
    
class Mesh(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype = np.float).view(cls)
        return obj    
        
def scale(pos, mesh):
    xmin, ymin, scale = mesh[0], mesh[2], float(mesh[4])
    x = pos[0] * scale + xmin
    y = pos[1] * scale + ymin
    return x, y

def histBins(positions, mesh):
    minX, maxX, minY, maxY, gridsize = mesh
    Xrange = np.arange(minX, maxX, gridsize)
    Yrange = np.arange(minY, maxY, gridsize)
    return Xrange, Yrange

def reMesh(dens1, dens2):
    if type(dens1) == Density and type(dens2) == Density:
        if not np.array_equal(dens1.mesh, dens2.mesh):
            dens2 = dens2.reMesh(dens1.mesh, copy=True)
        dens2 = dens2.view(np.ndarray)   
    return dens2

def densityMultiply(dens1, dens2):
    dens2 = reMesh(dens1, dens2)
    return dens1 * dens2

def densityAdd(dens1, dens2):
    dens2 = reMesh(dens1, dens2)
    return dens1 + dens2

def densityDivide(dens1, dens2):
    dens2 = reMesh(dens1, dens2)
    return dens1 / dens2

class NumberField(np.ndarray):
    
    def __new__(cls, positions, cellStates, mesh, types = None, cellType=None):
        xedge, yedge = histBins(positions, mesh)
        
        xlen = len(xedge) - 1
        ylen = len(yedge) - 1
        
        obj = np.zeros((xlen, ylen), dtype=np.float64)
        
        if cellType is not None and types is not None:
            positions = positions[types==cellType]
            cellStates = cellStates[types==cellType]
            
        Xpos = positions[:,0]
        Ypos = positions[:,1]  
        
        Xi = np.digitize(Xpos, xedge)
        Yi = np.digitize(Ypos, yedge)
        
        def addVal(xindex, yindex, val):
            obj[xindex-1, yindex-1] += val
            
        map(addVal, Xi, Yi, cellStates)
        return obj.view(cls)
    
    def __init__(self, positions, cellStates, mesh, types = None, cellType=None):
        if cellType is not None and types is not None:
            positions = positions[types==cellType]
            cellStates = cellStates[types==cellType]
            types = types[types==cellType]
        self.positions = positions
        self.cellStates = cellStates
        self.types = type
        self.cellType = cellType
        self.mesh = mesh
    
    def reMesh(self, mesh, copy=True):
        if copy:
            return numberField(self.positions, self.cellStates, mesh, self.types, self.cellType)
        else:
            self.__new__(self.positions, self.cellStates, mesh, self.types, self.cellType)  
            return self

class Density(np.ndarray):
    
    def __new__(cls, positions, mesh, types = None, cellType = None):
        edges = histBins(positions, mesh)
        
        if cellType is not None and types is not None:
            positions = positions[types==cellType]
            
        Xpos = positions[:,0]
        Ypos = positions[:,1]  
        hist, xedge, yedge = np.histogram2d(Xpos, Ypos, edges)
        
        return np.asarray(hist, dtype = np.int32).view(cls)
    
    def __init__(self, positions, mesh, types = None, cellType = None):
        if cellType is not None and types is not None:
            positions = positions[types==cellType]
            types = types[types==cellType]
        self.positions = positions
        self.types = types
        self.mesh = mesh
        self.cellType = cellType
        
    def reMesh(self, mesh, copy=True):
        if copy:
            return Density(self.positions, mesh, self.types, self.cellType)
        else:
            self.__new__(self.positions, mesh, self.types, self.cellType)  
            return self

        
class Contours(list):
    
    def __init__(self, numberField, threshold, mesh):
        self.mesh = mesh
        self.scale = mesh[4]
        scale = 256. / (numberField.max() - numberField.min())
        threshold = threshold * scale
        numberfield = (numberField - numberField.min()) * scale
        im = np.array(numberField, dtype=np.uint8)
        ret, thresh = cv2.threshold(im,threshold,256,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        self.hierarchy = hierarchy
        self[:] = map(lambda contour: Contour(contour, mesh), contours)
        
    def plot(self, show=False):
        import matplotlib.pyplot as plt
        for contour in self:
            plt.plot(contour[:,0,0], contour[:,0,1])
        if show:
            plt.show()    
    
    def areas(self):
        return map(lambda contour: contour.area(), self)
    
    def perimeters(self):
        return map(lambda contour: contour.perimeter(), self)
    
    def sumPerimeters(self):
        return sum(self.perimeters())
    
    def largestContour(self):
        maxI = np.argmax(self.areas())
        return self[maxI]

class Contour(np.ndarray):
    
    def __new__(cls, input_array, mesh):
        obj = np.asarray(input_array, dtype = np.int32).view(cls)
        return obj
    
    def __init__(self, input_array, mesh):
        self.mesh = mesh
        self.scale = float(mesh[4])
        
    def perimeter(self):
        return cv2.arcLength(self, True) * self.scale
    
    def area(self):
        return cv2.contourArea(self) * self.scale **2
    
    def moments(self):
        return cv2.moments(self), self.mesh
    
    def convexHull(self):
        return Contour(cv2.convexHull(self), self.mesh)
    
    def encloseRectangle(self):
        pos, (w, h), angle = cv2.minAreaRect(self)
        (x, y) = scale(pos, self.mesh)
        w, h  = w*self.scale, h*self.scale
        return (x, y), (w, h), angle

    def encloseCircle(self):
        pos,radius = cv2.minEnclosingCircle(self)
        (x,y) = scale(pos, self.mesh)
        radius *= self.mesh
        return (x,y), radius
    
    def fitEllipse(self):
        #Returns rotated rectangle which encloses ellipse
        pos, (w, h), angle = cv2.fitEllipse(self)
        (x, y) = scale(pos, self.mesh)
        w, h  = w*self.scale, h*self.scale
        return (x, y), (w, h), angle
        
class State(np.ndarray):
    
    def __new__(cls, cellStates=None, mesh=None, file=None):
        stateNames = ['number', 'type', 'radius', 'length',
                      'position', 'direction', 'mesh' ]
        stateFormats = ['u4', 'u1', 'f4', 'f4', 
                        'f4', 'f4', 'f4', 'f4', 'f4']  
        
        if cellStates is not None:
            N = len(cellStates)
        elif file is not None:
            with open(file, 'rb') as f:
                data = cPickle.load(f)
                N = len(data[0])

        shapes = [1, N, N, N, (N, 3), (N, 3), 5]
        formats = zip(stateFormats, shapes)
        statedtype = dict(names=stateNames, formats=formats)
        state = np.empty((1,), dtype=statedtype)
        return state.view(cls)
    
    def __init__(self, cellStates=None, mesh=None, file=None):
        if cellStates is not None:
            self.loadState(cellStates, mesh)
        elif file is not None:
            self.loadFile(file, mesh)
        
    def loadState(self, cellStates, mesh=None):
        
        def returnCellState(cellStates):
            cs = cellStates
            return cs.cellType, cs.radius, cs.length, cs.pos, cs.dir,
        
        N = len(cellStates)
        states = map(returnCellState, cellStates)
        
        if mesh is None:
            maxL = max([max(np.abs(cellState.pos)) for cellState in cellStates]) * 1.05
            stepL =  2 * maxL / 50
            mesh = (-maxL, maxL, -maxL, maxL, stepL,)
            
        self[0] = tuple([N,] + zip(*states) + [mesh,])
    
    def loadFile(self, pickleFile, mesh=None):
        '''
        loads state from pickle file
        '''
        with open(pickleFile, 'rb') as f:
            data = cPickle.load(f)
        cellStates = data[0].values()
        self.loadState(cellStates, mesh)
        
    def evalCharacteristic(self, mapping):
        return mapping(*self[0])

class Description:
    '''
    This the class in which a propagation of the reference is stored
    '''
    
    def __init__(self, N=None, states=None, mesh=None, folder=None):
        if states is not None:
            self.loadStates(states, mesh)
        elif folder is not None:
            self.loadFolder(folder)
            
    def loadFolder(self, folder):
        self.folder = folder
        self.pickleFiles = [picklefile for picklefile in os.listdir(folder) if picklefile.endswith('pickle')]
        self.times = [int(picklefile.split('-')[1].split('.')[0]) for picklefile in self.pickleFiles]
        self.states = [self.loadFile(picklefile) for picklefile in self.pickleFiles]
        data = zip(self.times, self.states)
        data.sort(lambda x, y: cmp(x[0],y[0]))
        self.times, self.states = zip(*data)
        
    def loadFile(self, pickleFile):
        os.chdir(self.folder)
        with open(pickleFile, 'rb') as f:
            data = cPickle.load(f)
        cellStates = data[0].values()
        state = State(cellStates)
        return state
    
    def evalCharacteristic(self, mapping):
        tlist = self.times
        names = ['time', 'characteristic']
        formats = ['float64', 'float64']
        dtype = dict(names=names, formats=formats)
        
        def extractArgs(state):
            names = state.dtype.names
            arg = map(lambda x: state[x], names)
            return arg

        characlist = map(lambda state: state.evalCharacteristic(mapping), self.states)
        
        charac = np.array(zip(tlist, characlist), dtype=dtype)
        return charac
           
class Domain(tuple):
    '''
    This defines the object where all the different propagation runs are stored
    The propagations are stored as the values of a dictionary.
    The keys of the dictionary are tuples of the folder and start descriptions
    The values are the list of runs of the reference from this starting point
    
    self.importFolder runs importRun on every folder inside the specified folder
    
    '''
    
    def __new__(cls, domainFolder):
        workDir = os.getcwd()
        os.chdir(domainFolder)
        folders = [folder for folder in os.listdir(domainFolder) if os.path.isdir(folder)]
        def convertFolder(folder):
            pickleFolders = os.listdir(folder)
            pickleFolders = map(lambda dir: os.path.join(domainFolder, folder, dir), pickleFolders)
            return tuple(map(lambda x: Description(folder=x), pickleFolders))
        
        newDomain = map(convertFolder, folders)
        os.chdir(workDir)
        return super(Domain, cls).__new__(cls, newDomain)
    
    def __init__(self, domainFolder):
        workDir = os.getcwd()
        os.chdir(domainFolder)
        self.folders = [folder for folder in os.listdir(domainFolder) if os.path.isdir(folder)]
        os.chdir(workDir)
        
    def extract(self):
        return tuple(self), self.folders
        
    def importDomain(self, domain):
        if len(domain.keys()) != len(self):
            self.__new__(domain)
        
        folders = domain.keys()

        for i, folder in enumerate(folders):
            run = map(Description, domain[folder])
            sim = np.empty(len(run), dtype='object')
            sim[:] = run
            self[i] = (folder, sim)
            
    def evalCharacteristics(self, mapping):
        names = ['folder', 'time', 'characteristic']
        nsim = len(self[0])
        #assert filter(lambda x, y: x == y * x, map(len, self)), 'Number of simulations not constant'
        nfolder = len(self.folders)
        time = self[0][0]['time']
        ntime = len(time)
        shape = (nsim, ntime)
        formats = ['a5', ('float64', ntime), ('float64', shape)]
        dtype = dict(names=names, formats=formats)
        charac = np.empty(nfolder, dtype=dtype)
        
        def mapSim(sim, mapping):
            evalSim = map(lambda x: x.evalCharacteristic(mapping)['characteristic'], sim)
#                 charac = np.array([run['characteristic'] for run in evalSim])
#                 time = [run['time'] for run in evalSim]
            return evalSim
        
        charac['characteristic'] = map(lambda x: mapSim(x, mapping), self)
        charac['time'] = time
        charac['folder'] = self.folders
        
        return charac
    
if __name__ == '__main__':   
    pickleFile = 'Domain/example.pkl'
    folder = '/home/mg542/Source/HOTBOT'
    os.chdir(folder)
    state = State(file=pickleFile)
    position = state[0]['position']
    cellTypes = state[0]['type']
    dir = state[0]['direction']
    
    
    
