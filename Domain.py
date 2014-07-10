'''
Created on 21 May 2014

@author: mg542
'''

import pickle
import os
import operator

import numpy as np

from Description import Description

directory = '/scratch/mg542/Data/CellModeller'
names = ['time', 'state']
formats = ['float64', 'object']
dtype = dict(names=names, formats=formats)

mapping = lambda x1, x2, x3, x4, x5, x6, x7, x8: np.mean(x4)           

def domainTuple(domain):
    foldersNames = domain.keys()
    DomainTup = namedtuple('DomainTup', foldersNames, verbose=True, rename=True)
    
    def convertFolder(folder):
        return tuple(map(Description, domain[folder]))
        
    newDomain = map(convertFolder, foldersNames)
    
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
