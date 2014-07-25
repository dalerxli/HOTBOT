
import os
import cPickle
import random
import atexit
from datetime import datetime
from collections import OrderedDict

import numpy as np

from deap import creator, base, gp

from Primitives import *
from storePrimitives import *
from Description import *


domainFile = '/home/mg542/Data/CellModeller/domain.pkl'
pset = loadPset('/home/mg542/Source/HOTBOT/pset.pkl')
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0, -1.0))
creator.create("Robustness", float)
creator.create("Shift", float)
creator.create("Characteristics", OrderedDict, timeSeries=None)
creator.create("DomainSD", float)
creator.create("DomainAv", float)
creator.create("SVDcoef", float)
creator.create("Label", int)
creator.create("Size", int)
creator.create("Individual", gp.PrimitiveTree,
               fitness=creator.FitnessMulti, pset=pset,
               charac=creator.Characteristics,
               robustness=creator.Robustness,
               shift=creator.Shift,
               domainSD=creator.DomainSD,
               domainAv=creator.DomainAv,
               label=creator.Label,
               size=creator.Size,
               svd=creator.SVDcoef)


def cleanup():
    print "Goodbye"
atexit.register(cleanup)

store = "/sharedscratch/mg542/store/"

def findMostRecent(folder=store):
    fmt = '%Y-%m-%dT%H:%M:%S'
    os.chdir(store)
    runs = [folder for folder in os.listdir(store) if os.path.isdir(folder) and folder != 'profile']
    #times = [datetime.strptime(folder, fmt) for folder in runs]
    times = [os.path.getmtime(folder) for folder in runs]
    mostRecent = runs[times.index(max(times))]
    mostRecent = os.path.join(store, mostRecent)
    #mostRecent = '/sharedscratch/mg542/store/2014-07-11T10:25:33'

    pickles = [file for file in os.listdir(mostRecent) if file.endswith('.pkl')]
    picklePaths = [os.path.join(mostRecent, file) for file in pickles]
    creationTimes = [os.path.getctime(file) for file in picklePaths]
    mostRecentFile = picklePaths[creationTimes.index(max(creationTimes))]
    return mostRecentFile

def similar(ind1, ind2):
    charac1 = ind1.charac['characteristic']
    charac2 = ind2.charac['characteristic']
    diff = charac1/charac2
    std = diff.std()
    if std == 0:
        return True
    else:
        return False


def removeDuplicates(population):
    unique = population[:]
    for ind1 in population:
        has_twin = False
        to_remove = []
        for ind2 in unique:    # hofer = hall of famer
            if ind1 == ind2:
                has_twin = True
                to_remove.append(ind2)
        
        for ind in reversed(to_remove):       # Remove the dominated hofer
            unique.remove(ind)
        if has_twin:
            unique.append(ind1)
    return unique

def plotCharac(charac):
    import matplotlib.pyplot as plt
    values = charac['characteristic']
    for sim in values:
        plt.figure()
        for run in sim:
            plt.plot(run)
    plt.show()
    
def plotCharacSingle(charac):
    import matplotlib.pyplot as plt
    times, values = charac['time'], charac['characteristic']
    for time, sim in zip(times, values):
        error = sim.std(axis=0)
        average = sim.mean(axis=0)
        plt.errorbar(time, average, yerr=error)
    plt.show()
    
def shift(ind):
    return ind.shift/ind.robustness

def robust(ind):
    return ind.robustness/ind.domainSD

def evalSVD(atlas):
    def flatten(individual):
        charac = individual.charac['characteristic'].flatten()
        return charac/(individual.domainSD + np.abs(individual.domainAv))
    M = np.vstack(map(flatten, atlas)).T
    U, s, V = np.linalg.svd(M)
    return U, s, V
    
if __name__ == '__main__':
    
#    with open(domainFile, 'rb') as f:
        #domain = cPickle.load(f)
    pass

    pickle = '/home/mg542/8-15:55:20.pkl' #findMostRecent(store)

    directory, pickleFile = os.path.split(pickle)

    print directory, pickleFile
    print os.listdir(directory)


    with open(pickle, 'rb') as f:
        cp = cPickle.load(f)

    atlases = cp['atlases']
    logbook = cp['logbook']
    atlas = sum(atlases, [])
    import time
    t0 = time.time()
    unique = removeDuplicates(atlas)
    t1 = time.time()

    robustInd = [ind for ind in atlas if ind.robustness/ind.domainSD < 0.3]
    robustList = [ind.robustness/ind.domainSD for ind in robustInd]    
    
    good = [ind for ind in atlas if all((shift(ind) > 2, robust(ind) < 0.3))]
