
import operator
import random
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

import scoop.futures as futures

from Primitives import *
from storePrimitives import *
from Domain2 import DomainTuple
from Description import Description, Domain
from GeneticProgramming import *

#loading domain
extractFile = '/scratch/mg542/Simulations/extract2.pkl'
with open(extractFile, 'rb') as f:
    extract = pickle.load(f)

domain = DomainTuple(domain=None, extract=extract)


directory = '/scratch/mg542/Simulations'

store = '/scratch/mg542/store'
psetFile = '/home/mg542/Source/HOTBOT/pset.pkl'
atlasesFile = '/scratch/mg542/store/2014-06-23T12:21:08/10:01:37.pkl'
atlasesFile = None

#Loading Primitives
pset = loadPset(psetFile)

nAtlases = 2
nSwaps = 3
nMappings = 300
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
minDiv = (0.6, 500) #1e-2


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

def evalCharacteristic(individual):
    '''
    This function applies a mapping to the domain of propagations
    returning the domain of characteristics
    '''
    mapping = gp.compile(individual, pset)
    domainCharac = domain.evalCharacteristics(mapping)
    return domainCharac

def selectSVD(atlas, minDiv=1e-2):
    atlas.sort(lambda x, y: cmp(x.shift, y.shift))
    U, s, V = evalSVD(atlas)
    
    for i, ind in enumerate(atlas):
        ind.label = i
    
    def selParetoRank(sValue, vValues):
        if sValue < 1e-5 :
            return []
        
        def calcFitness(uVal, ind):
            shift, robust = ind.shift, ind.robustness
            norm = np.abs(uVal)
            if norm < minDiv:
                ind.fitness.values = 0., np.inf, 0., len(ind)
            elif not np.isfinite(shift):
                ind.fitness.values = 0., np.inf, 0., len(ind)
            else:
                ind.fitness.values = shift*norm, robust/norm, norm, len(ind)
            ind.svd = norm
            ind.size = len(ind)
        map(calcFitness, vValues, atlas)            

        nRet = 5
        if len(atlas) < nRet:
            nRet = len(atlas)
        return tools.selSPEA2(atlas, 5)
        #return returnParetoFront(atlas)
    
    selected = sum(map(selParetoRank, s, V), [])
    
    #Discard Duplicates
    removed = []
    for ind in selected:
        addInd = True
        for ind2 in removed:
            if ind2.label == ind.label:
                addInd = False
        if addInd:
            removed.append(ind)
    
    return removed

toolbox = base.Toolbox()
toolbox.register('map', map)
toolbox.register('multiMap', futures.map)

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
toolbox.register("domainAv", evalDomainAv)
toolbox.register("evalCRS", evalCRSSA, toolbox=toolbox)
toolbox.register("evalDiversity", evalDiversity)
toolbox.register("selSVD", selectSVD)
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
    
    stats = tools.Statistics(lambda ind: (ind.shift, ind.robustness, ind.svd, ind.size,))#ind.fitness.values)
    stats.register("Atlas", lambda stat: map(lambda x: 
                                             ("%.0f"%np.log10(x[0]), "%.0f"%np.log10(x[1]),
                                              #"%.2e" % x[0], "%.2e" % x[1],
                                              "%.1f"%x[2], x[3])
                                             # "%.1e" % x[2], "%.1e" % x[3],
                                             # "%.i" % x[4]) 
                                             # map(lambda y: "%.1e" % y, x)
                                                 , stat))
    
    atlases, logbook = eaParetosSVD(atlases, toolbox, cxpb, mutpb, ngen, 
                                 minDiv, stats=stats, #halloffame=hof,
                                 checkpoint=folder, freq=freq)
    return atlases, logbook#, hof
            
        
if __name__ == '__main__':
    
    atlases, logbook = paretos(checkpoint=store, freq=checkFreq)
