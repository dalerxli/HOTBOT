'''
Created on 21 May 2014

@author: mg542
'''

from collections import OrderedDict, namedtuple
import pickle
import os
import operator

import numpy as np

from Description import Description


directory = '/scratch/mg542/Simulations'

names = ['time', 'state']
formats = ['float64', 'object']
dtype = dict(names=names, formats=formats)

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


        

mapping = lambda x1, x2, x3, x4, x5, x6, x7, x8: np.mean(x2)           

def domainTuple(domain):
    foldersNames = domain.keys()
    DomainTup = namedtuple('DomainTup', foldersNames, verbose=True, rename=True)
    
    def convertFolder(folder):
        return tuple(map(Description, domain[folder]))
        
    newDomain = map(convertFolder, foldersNames)
    
class DomainTuple(tuple):
    '''
    This defines the object where all the different propagation runs are stored
    The propagations are stored as the values of a dictionary.
    The keys of the dictionary are tuples of the folder and start descriptions
    The values are the list of runs of the reference from this starting point
    
    self.importFolder runs importRun on every folder inside the specified folder
    
    '''
    
    def __new__(cls, domain=None, extract=None):
        if extract is None:
            folders = domain.keys()
            def convertFolder(folder):
                return tuple(map(Description, domain[folder]))
            newDomain = map(convertFolder, folders)
        else:
            newDomain = extract[0]
        self = super(DomainTuple, cls).__new__(cls, newDomain)
        return self
    
    def __init__(self, domain, extract=None):
        if extract is None:
            self.folders = domain.keys()
        else:
            self.folders = extract[1]
        
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

class DomainNP(np.ndarray):
    '''
    This defines the object where all the different propagation runs are stored
    The propagations are stored as the values of a dictionary.
    The keys of the dictionary are tuples of the folder and start descriptions
    The values are the list of runs of the reference from this starting point
    
    self.importFolder runs importRun on every folder inside the specified folder
    
    '''
    names = ['folder', 'sim']
    formats = ['a5', 'object']
    
    def __new__(cls, domain = None):
        if domain is not None:
            length = len(domain.keys())            
        dtype = dict(names=cls.names, formats=cls.formats)
        obj = np.empty(length, dtype = dtype)
        obj = np.asarray(obj).view(cls)
        return obj.view(cls)        

    def __init__(self, domain = None):
        if domain is not None:
            self.importDomain(domain)
        
    def importDomain(self, domain):
        if len(domain.keys()) != len(self):
            self.__new__(domain)
        
        folders = domain.keys()
        
        for i, folder in enumerate(folders):
            run = map(Description, domain[folder])
            sim = np.empty(len(run)  , dtype='object')
            sim[:] = run
            self[i] = (folder, sim)
            
    def evalCharacteristic(self, mapping):
        names = ['folder', 'time', 'characteristic']
        nsim = len(self['sim'][0])
        nfolder = len(self['folder'])
        time = self['sim'][0][0]['time']
        ntime = len(time)
        shape = (nsim, ntime)
        formats = ['a5', ('float64', ntime), ('float64', shape)]
        dtype = dict(names=names, formats=formats)
        charac = np.empty(nfolder, dtype=dtype)
        
        def mapSim(sim, mapping):
            evalSim = map(lambda x: x.evalCharacteristic(mapping), sim)
            charac = np.array([run['characteristic'] for run in evalSim])
            time = np.array([run['time'] for run in evalSim])
            return charac, time
        
        charac['characteristic'] = map(lambda x: mapSim(x, mapping)[0], self['sim'])
        charac['time'] = map(lambda x: mapSim(x, mapping)[1][0], self['sim'])
        charac['folder'] = self['folder']
        
        return charac

if __name__ == '__main__':
    os.chdir(directory)
    with open('domain.pkl', 'rb') as f:
        domain = pickle.load( f)
    
    with open('extract.pkl', 'wb') as f:
        pickle.dump(domain.extract(), f)
        
    with open('np.pkl', 'wb') as f:
        pickle.dump(domain[0][0], f)
    
    newDomain = []
    for folder, sim in zip(domain.folders, domain):
        import NewDescription
        os.chdir(directory)
        with open(folder + '/Larr.pkl', 'rb') as f:
            Larr = pickle.load(f)    
            print folder, Larr[0], len(sim)
            sim2 = []
            for ref in sim:
                ref2 = NewDescription.Description(ref, Larr[0])
                print ref2['state'][0].dtype.names
                sim2.append(ref2)
            sim2 = tuple(sim2)
        newDomain.append(sim2)
        
    newDomain2=DomainTuple(domain=None, extract=((newDomain, domain.folders)))
    
    with open('extract2.pkl', 'wb') as f:
        pickle.dump((newDomain, domain.folders), f)
                
    #domain = Domain()
    #domain.importRun('93')
