'''
Created on 21 May 2014

@author: mg542


Data Structures:

    An atlas is a list of individuals:
    atlas = [ind,]
    
    atlases is a list of different atlases:
    atlases = [atlas,]
    
    An individual - ind - is of type primitive and has various attributes:
    func = gp.compile(ind, pset)
    fitness = ind.fitness.values
    robustness = ind.robustness
    charac = ind.characteristics
    timeSeries = ind.characteristics.timeSeries
    
    mapping = toolbox.compile(ind, pset)

Propagations are stored as Description Objects.

    propagation = Description(listMass, listRadius,
                              listPositions, listMomementa, 
                              fieldBox, XYZ, time = None):
                          
The characteristic of each timestep of the propagation
of a mapping are found by calling the method:

    charac, timesteps = propagation.evalCharacteristic(mapping)
    
The Domain of propagations are stored in the Domain Object.

    domain = domain(folder)
    
The domain stores the propagation in a defaultdict.
The keys of the defaultdict store the subfoldername and starting description

    [Description,] = domain[(subfoldername, intDescription)]
    
    domainCharac = domain.evalCharaceristics(mapping)
    
    [Characteristics,] = domainCharac[(subfoldername, intCharacteristc)]
    
    Charcteristics = ([characteristic, ], timeSeries])
    
    domainRep = domain.evalRepresentation(atlas)
    
    domainCharac = domainRep[ind]
    
                 


'''



import operator
import random
import math
import pickle
import os
from datetime import datetime
from collections import OrderedDict

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools

toolbox = base.Toolbox()


# from scoop.futures import map as map_
# from scoop import shared
# import scoop
# from scoop.futures import map_as_completed
import scoop.futures as futures
toolbox.register('map', map)
toolbox.register('multiMap', futures.map)
toolbox.register('map_as', map_as_completed)


from storePrimitives import *
from Primitives import *
from Domain2 import DomainTuple
from NewDescription import Description
from GeneticProgramming import *

#loading domain
extractFile = '/scratch/mg542/Simulations/extract2.pkl'
with open(extractFile, 'rb') as f:
    extract = pickle.load(f)

domain = DomainTuple(domain=None, extract=extract)


directory = '/scratch/mg542/Simulations'

store = '/scratch/mg542/store'
psetFile = '/home/mg542/Documents/Source/Gas/pset.pkl'
atlasesFile = '/scratch/mg542/store/2014-06-23T12:21:08/10:01:37.pkl'
#atlasesFile = None

#Loading Primitives
pset = loadPset(psetFile)

nAtlases = 3
nSwaps = 3
nMappings = 10
ngen = 30000

checkFreq = 100

treeMin = 2
treeMax = 5

mutTreeMin = 0
mutTreeMax = 3
cxpb = 0.5
mutpbUniform = 0.1
mutpbShrink = 0.05
mutpbEphemeral = 0.4

#Adaptive diversity scaling
minDiv = (0.7, 3000) #1e-2


creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, -1.0, -1.0))
creator.create("Robustness", float)
creator.create("Shift", float)
creator.create("Characteristics", OrderedDict, timeSeries=None)
creator.create("domainSD", float)
creator.create("Individual", gp.PrimitiveTree,
               fitness=creator.FitnessMulti, pset=pset,
               charac=creator.Characteristics,
               robustness=creator.Robustness,
               shift=creator.Shift,
               domainSD=creator.domainSD)

def evalCharacteristic(individual):
    '''
    This function applies a mapping to the domain of propagations
    returning the domain of characteristics
    '''
    mapping = gp.compile(individual, pset)
    domainCharac = domain.evalCharacteristics(mapping)
    return domainCharac

# Creating population generation operators.
toolbox.register("expr", genHalfAndHalf, pset=pset, type_=pset.ret, min_=treeMin, max_=treeMax)
toolbox.register("expr_mut", genFull, min_=mutTreeMin, max_=mutTreeMax)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Creating evaluation operators
toolbox.register("characteristic", evalCharacteristic)#, domain=domain)
toolbox.register("robustness", evalRobustness)
toolbox.register("shift", evalShift)
toolbox.register("domainSD", evalDomainSD)
toolbox.register("evalCRS", evalCharacRobustShift, toolbox=toolbox)
toolbox.register("evalDiversity", evalDiversity)
toolbox.register("evalFitness", evalFitness, toolbox=toolbox)
toolbox.register("divRobust", evalDivFit, toolbox=toolbox, sort='robust')
toolbox.register("divShift", evalDivFit, toolbox=toolbox, sort='shift')
toolbox.register("evalPareto", evalPareto, toolbox=toolbox, similar=lambda x, y:equal(x, y, minDiv))
toolbox.register("pareto", returnPareto, similar=operator.eq)

# Creating mutations and mating operators
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutateEphemeral", gp.mutEphemeral, mode='one')
toolbox.register("mutateUniform", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mutateShrink", gp.mutShrink)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

mutpb = {toolbox.mutateEphemeral : mutpbEphemeral, 
         toolbox.mutateUniform : mutpbUniform, 
         toolbox.mutateShrink : mutpbShrink, }

def paretos(checkpoint=None, freq=50):
    
    if checkpoint is not None:
        os.chdir(checkpoint)
        foldername = datetime.today().replace(microsecond=0).isoformat()
        os.mkdir(foldername)
        os.chdir(foldername)
        folder = os.getcwd()
    
    if atlasesFile is not None:
        with open(atlasesFile, 'rb') as f:
            cp = pickle.load(f)
        atlases = cp['atlases']
        if len(cp['atlases']) > nAtlases:
            atlases = cp['atlases'][:nAtlases]
            for i, atlas in enumerate(cp['atlases'][nAtlases:]):
                atlases[i % nAtlases][:] = atlases[i % nAtlases] + atlas
        for atlas in atlases:
            atlas[:] = atlas + list(cp['constantMappings'])     
    else:
        atlases = []
        for i in range(nAtlases):
            atlases.append(toolbox.population(n=nMappings))
        
        for atlas in atlases:
            for ind in atlas:
                ind.charac = None
                ind.robustness = None
                ind.shift = None
        
    #hof = tools.HallOfFame()
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Atlas", lambda stat: map(lambda x: 
                                             ("%.2f" % x[0], "%.2f" % x[1],
                                              "%.1e" % x[2], "%.1e" % x[3],
                                              "%.i" % x[4]) 
                                             # map(lambda y: "%.1e" % y, x)
                                                 , stat))
    
    atlases, logbook = eaParetos(atlases, toolbox, cxpb, mutpb, ngen, 
                                 minDiv, stats=stats, #halloffame=hof,
                                 checkpoint=folder, freq=freq)
    return atlases, logbook#, hof
            
        
if __name__ == '__main__':
    
    atlases, logbook = paretos(checkpoint=store, freq=checkFreq)
