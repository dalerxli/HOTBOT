
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

if __name__ == '__main__':
	
	with open(domainFile, 'rb') as f:
		#domain = cPickle.load(f)
		pass

	pickle = findMostRecent(store)

	directory, pickleFile = os.path.split(pickle)

	print directory, pickleFile
	print os.listdir(directory)


	with open(pickle, 'rb') as f:
		cp = cPickle.load(f)

	atlases = cp['atlases']
	logbook = cp['logbook']
	atlas = sum(atlases, [])
	unique = removeDuplicates(atlas)
	robustInd = [ind for ind in unique if ind.robustness/ind.domainSD < 0.3]
	robustList = [ind.robustness/ind.domainSD for ind in robustInd]	
	good = [ind for ind in robustInd if ind.shift/ind.robustness > 1.5]	
	
	print len(robustInd), len(atlas)
	print pickle
	print 'evals', logbook[-1]['nevals']
	print logbook[-1]['natlas']
	print 'gen', logbook[-1]['gen']
	print 'good', len(good)
