'''
Created on 20 May 2014

@author: mg542
'''

import warnings
import random
import sys
import operator
import os
import pickle
from datetime import datetime
from inspect import isclass

from progressbar import ETA, ProgressBar, Percentage, Bar
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools

def evalCharacteristic(individual, domain, toolbox, out=False):
    '''
    This function applies a mapping to the domain of propagations
    returning the domain of characteristics
    '''
    mapping = toolbox.compile(individual)
    domainCharac = domain.evalCharacteristics(mapping)
    #individual.charac = domainCharac
    #if out:
    return domainCharac

def evalDomainSD(individual):
    '''
    This calculates the standard deviation of a characteristic over 
    the entire domain. This allows the shift and robustness to be normalised
    '''
    charac = individual.charac['characteristic']
    normSD = charac.std()
    
    individual.domainSD = normSD
    return normSD
    
def evalDomainAv(individual):
    charac = individual.charac['characteristic']
    domainAv = charac.mean()
    
    individual.domainAv = domainAv
    return domainAv

def evalShift(individual):
    '''
    This evaluates how much a characteristic changes during propagation.
    '''
    
    charac = individual.charac['characteristic']
    #domainCharac = np.concatenate(charac.values())
    shifts = charac.std(axis = 2)
    shift = shifts.mean()
    individual.shift = shift
            
    return shift
    
def evalRobustness(individual):
    '''
    This calculates the fitness of a characteristic
    It does this by comparing the standard deviation of groupings of 
    propagations to the standard deviation of the domain.
    '''
    charac = individual.charac['characteristic']

    
    #calculating sd of characteristics across a set of simulations
    #during propagation.
    robustness = charac.std(axis = 1)
    robustness = np.mean(robustness)
    individual.robustness = robustness
            
    return robustness

def evalCharacRobustShift(atlas, toolbox):
    invalid_ind = [ind for ind in atlas if ind.charac is None]
    
    charac = list(toolbox.multiMap(toolbox.characteristic, invalid_ind))
    for ind, C in zip(invalid_ind, charac):
        ind.charac = C
    
    SD = list(map(toolbox.domainSD, invalid_ind))
    robust = list(map(toolbox.robustness, invalid_ind))
    shift = list(map(toolbox.shift, invalid_ind))
    
    return len(invalid_ind)
    
def evalCRSSA(atlas, toolbox):
    invalid_ind = [ind for ind in atlas if ind.charac is None]
    
    charac = list(toolbox.multiMap(toolbox.characteristic, invalid_ind))
    for ind, C in zip(invalid_ind, charac):
        ind.charac = C
    
    Av = list(map(toolbox.domainAv, invalid_ind))
    SD = list(map(toolbox.domainSD, invalid_ind))
    robust = list(map(toolbox.robustness, invalid_ind))
    shift = list(map(toolbox.shift, invalid_ind))
    
    return len(invalid_ind)

def evalDiversity(atlas, sort = 'robust'):
    '''
    This calculates a correlation matrix of the characteristics
    This is then used to weight against similar mappings
    '''
    if sort == 'robust':
        atlas.sort(lambda x, y: cmp(x.robustness, y.robustness))
    elif sort == 'shift':
        atlas.sort(lambda x, y: cmp(x.shift, y.shift))
    #This generates a flattened list of characteristics
    #Of a domain generated by a mapping.
    def flatten(individual):
        return individual.charac['characteristic'].flatten()
    
    #Generate Array of characteristics.
    M = np.vstack(map(flatten, atlas))
    
    #Return correlation matrix
    Corr = np.corrcoef(M)
    diversityMatrix = np.abs(np.tril(Corr, k = -1))
    return diversityMatrix

def evalSVD(atlas):#, sort = 'robust'):
    '''
    This calculates a correlation matrix of the characteristics
    This is then used to weight against similar mappings
    '''
#     if sort == 'robust':
#         atlas.sort(lambda x, y: cmp(x.robustness, y.robustness))
#     elif sort == 'shift':
#         atlas.sort(lambda x, y: cmp(x.shift, y.shift))
    
    #This generates a flattened list of characteristics
    #Of a domain of characteristics generated by a mapping.
    # this is then normalised
    def flatten(individual):
        charac = individual.charac['characteristic'].flatten()
        return charac/(individual.domainSD + np.abs(individual.domainAv))
    
    #Generate Array of characteristics.
    M = np.vstack(map(flatten, atlas)).T
    
    #Return correlation matrix
    U, s, V = np.linalg.svd(M)
    return U, s, V
    
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
#         for uVal, ind in zip(vValues, atlas):
#             shift, robust = ind.shift, ind.robustness
#             norm = np.abs(uVal)
#             ind.fitness.values = shift*norm, robust*norm, len(ind)
#             ind.svd = uVal
#             ind.size = len(ind)
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
    
def evalDivFit(atlas, toolbox, sort = 'robust'):
    
    diversity = np.nan_to_num(toolbox.evalDiversity(atlas, sort))
    fitness = map(np.prod, 1-diversity)
                
    return fitness

def equal(ind1, ind2, minDiv = 1e-3):
    if operator.eq(ind1, ind2):
        return True
    elif ind1.fitness.values[0] < minDiv:
        return True
    elif ind1.fitness.values[1] < minDiv:
        return True
    else:
        return False    
        
def evalPareto(atlas, toolbox, similar=None):
    pareto = tools.ParetoFront(similar)
    diversityRobust = toolbox.divRobust(atlas)
    diversityShift = toolbox.divShift(atlas)
    
    fitnesses = toolbox.map(lambda ind, R, S: [R, S, ind.shift, ind.robustness], 
                            atlas, diversityRobust, diversityShift)
    
    for ind, fit in zip(atlas, fitnesses):
        #replacing invalid fitness values
        fit[2] /= ind.domainSD
        fit[3] /= ind.domainSD
        if sum(map(np.isfinite, fit)) !=  len(fit):
            fit = [-1, -1, -1, np.inf, np.inf]
        elif fit[0] > 1 or fit[1] > 1:
            fit = [-1, -1, -1, np.inf, np.inf]
        ind.fitness.values = fit
    
    pareto.update(atlas)
    return pareto

def evalFitness(atlas, toolbox):
    #pareto = tools.ParetoFront(similar)
    diversityRobust = toolbox.divRobust(atlas)
    diversityShift = toolbox.divShift(atlas)
    
    fitnesses = toolbox.map(lambda ind, R, S: [R, S, ind.shift, ind.robustness, len(ind)], 
                            atlas, diversityRobust, diversityShift)
    
    for ind, fit in zip(atlas, fitnesses):
        #replacing invalid fitness values
        fit[2] /= ind.domainSD
        fit[3] /= ind.domainSD
        if sum(map(np.isfinite, fit)) !=  len(fit):
            fit = [-1, -1, -1, np.inf, np.inf]
        ind.fitness.values = fit
    
    return atlas

def returnPareto(atlas, similar=operator.eq):
    pareto = tools.ParetoFront(similar)
    pareto.update(atlas)   
    return pareto

def eaParetosSVD(atlases, toolbox, cxpb, mutpb, ngen, minDiv,
               minSize=20, stats=None, halloffame=None,
               checkpoint = None, freq = 50,  
               verbose=__debug__):
    """
    Structure of this algorithm:
    
    Calculates fitness of mappings in atlas
    """
    
    widgets = ['Evolving: ', Percentage(), ' ', 
               Bar(marker='#', left='[',right=']'),
               ' ', ETA()]
    
    nevals = 0
    paretoN = 0
    adaptiveMinDiv = False
    lastcheckpoint = None
    
    logbook = tools.Logbook()
    logbook.header = (stats.fields if stats else []) + ['gen', 'nevals','natlas']

    #Generate population of migrants to allow transfer between atlases
    migrants = []
    
    #
    constantMappings = ConstantMaps(operator.eq)
    
    pbar = ProgressBar(widgets=widgets, maxval=ngen+1)
    pbar.start() 
    # Begin the generational process
    for gen in range(1, ngen+1):
        
        #Evaluate fitness and pareto fronts for the generation
        #nevals += sum(toolbox.map(toolbox.evalCRS, atlases))
        nevals += toolbox.evalCRS(sum(atlases, []))
        
        #filter invalid mappings out so that SVD can be done:
        def filterInd(individual):
            validCharac = np.isfinite(individual.charac['characteristic'].sum())
            validNorm = not np.isclose(individual.domainSD + np.abs(individual.domainAv),0)
            validRobust = individual.robustness < 1e10
            return validCharac and validNorm and validNorm
        atlases = toolbox.map(lambda x: filter(filterInd, x), atlases)
       
        if type(minDiv) is tuple:
            a, b = minDiv
            adaptiveMinDiv = True
        if adaptiveMinDiv:
            minDiv = 1 - 10**( - paretoN**a / b) 
        
        selectedAtlases = toolbox.map(lambda atlas: selectSVD(atlas, minDiv), atlases)
        
        # Match size of migrants to pareto fronts
        paretoN = sum(map(len, selectedAtlases))
        while paretoN > len(migrants) or len(atlases) * minSize > len(migrants):
            ind = toolbox.individual()
            ind.charac = None
            ind.robustness = None
            ind.shift = None
            migrants.append(ind)
        
        #Place selected individuals in migrant population
        selected = sum(selectedAtlases, [])
        swap(selected, migrants)
        
        # Vary the pool of individuals    
        migrants = varAnd(migrants, toolbox, cxpb, mutpb)
        
        #Invalidate mutated individuals
        def newInd(ind):
            ind.charac = None
            ind.robustness = None
            ind.shift = None
            del ind.fitness.values
            return ind
        
        invalid_ind = [newInd(ind) for ind in migrants if not ind.fitness.valid]
        
        #create new atlases and mutate migrants:
        def nextGen(selected):
            if len(selected) > minSize:
                offspring = random.sample(migrants, len(selected))
            else:
                offspring = random.sample(migrants, 2*minSize)
            
            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(selected)
            
            return selected + offspring
        
        atlases =  toolbox.map(nextGen, selectedAtlases)
        

        record = stats.compile(sum(selectedAtlases, [])) if stats else {}
        logbook.record(gen=gen, nevals=nevals, natlas=map(len, selectedAtlases), **record)

        if verbose:
            print logbook.stream
            
        if checkpoint is not None and gen % freq == 0:
            os.chdir(checkpoint)
            filename = datetime.now().time().replace(microsecond=0).isoformat() + '.pkl'
            cp = dict(atlases = selectedAtlases,  generation=gen, 
                      halloffame=halloffame, logbook=logbook, 
                      migrants=migrants,
                      rndstate=random.getstate(),)
            with open(filename, 'wb') as f:
                pickle.dump(cp, f, -1)
            print "Checkpointed at generation %.i" % gen + ' in ' + checkpoint + '/' + filename
            lastcheckpoint = gen
        elif lastcheckpoint is not None:
            print "Last checkpointed at generation %.i" % lastcheckpoint + ' in ' + checkpoint + '/' + filename
        pbar.update(gen)
        print ''        
    pbar.finish()
            
    return paretos, logbook

def eaParetos(atlases, toolbox, cxpb, mutpb, ngen, minDiv,
               minSize=20, stats=None, halloffame=None,
               checkpoint = None, freq = 50,  
               verbose=__debug__):
    """
    Structure of this algorithm:
    
    Calculates fitness of mappings in atlas
    """
    
    widgets = ['Evolving: ', Percentage(), ' ', 
               Bar(marker='#', left='[',right=']'),
               ' ', ETA()]
    
    nevals = 0
    paretoN = 0
    adaptiveMinDiv = False
    lastcheckpoint = None
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'npareto', 'natlas', 'nconst'] + (stats.fields if stats else [])

    #Generate population of migrants to allow transfer between atlases
    migrants = []
    
    #
    constantMappings = ConstantMaps(operator.eq)
    
    pbar = ProgressBar(widgets=widgets, maxval=ngen+1)
    pbar.start() 
    # Begin the generational process
    for gen in range(1, ngen+1):
        
        #Evaluate fitness and pareto fronts for the generation
        #nevals += sum(toolbox.map(toolbox.evalCRS, atlases))
        nevals += toolbox.evalCRS(sum(atlases, []))
        
        atlases = list(toolbox.map(toolbox.evalFitness, atlases))
        
        def goodMap(ind):
            divS = ind.fitness.values[0] == 1
            divR = ind.fitness.values[1] == 1
            shift = ind.fitness.values[2] != 0
            robust = ind.fitness.values[3] < 1e10
            return divS and divR and shift and robust
        
        goodMaps = []
        otherMaps = []
        for atlas in atlases:
            goodMaps.append([ind for ind in atlas if goodMap(ind)])
            otherMaps.append([ind for ind in atlas if not goodMap(ind)])
        
        paretos = toolbox.map(toolbox.pareto, otherMaps)
        #paretos = toolbox.map(toolbox.evalPareto, atlases)
        
        # Match size of migrants to pareto fronts
        paretoN = sum(map(len, paretos + goodMaps))
        while paretoN > len(migrants) or len(atlases) * minSize > len(migrants):
            ind = toolbox.individual()
            ind.charac = None
            ind.robustness = None
            ind.shift = None
            migrants.append(ind)
            
        if type(minDiv) is tuple:
            a, b = minDiv
            adaptiveMinDiv = True
        if adaptiveMinDiv:
            minDiv = 1 - 10**( - paretoN**a / b)
        #print minDiv
        
        allSelected = []
        #create new atlases and mutate migrants:
        for pareto, atlas, goodM in zip(paretos, atlases, goodMaps):
            
            #create a list of all the elements in the pareto front
            #filtering all the constant mappings
            selected = [ind for ind in pareto if ind.fitness.values[3] != 0]
            constants = [ind for ind in pareto if ind.fitness.values[3] == 0]
            
            newConstants = constantMappings.update(constants)
            
            #filter out mappings which are too similar
            selected = [ind for ind in selected if ind.fitness.values[0] > minDiv 
                        or ind.fitness.values[1] > minDiv]
            
            #filter out mappings with bad robustness
            selected = [ind for ind in selected if ind.fitness.values[3] < 1e10]
            
            selected = goodM + selected
#             selected = filter(lambda y: y.fitness.values[0] > minDiv or 
#                               y.fitness.values[1] > minDiv, selected)
            
            #exchange individuals with the migrant population
            swap(selected + newConstants, migrants)
            
            #select offspring from population of migrants
            if len(selected) > minSize:
                offspring = random.sample(migrants, len(selected))
            else:
                offspring = random.sample(migrants, 2*minSize)
            
            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(selected)
            
            allSelected.append(selected)
            # Replace the current population by the offspring
            atlas[:] = selected + offspring
        
        # Vary the pool of individuals    
        migrants = varAnd(migrants, toolbox, cxpb, mutpb)
        #Invalidate mutated individuals
        invalid_ind = [ind for ind in migrants if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.charac = None
            ind.robustness = None
            ind.shift = None
            del ind.fitness.values
        
        
        record = stats.compile(sum(allSelected, [])) if stats else {}
        logbook.record(gen=gen, nevals=nevals, npareto=paretoN, natlas=map(len, allSelected),
                       nconst=len(constantMappings), **record)
        

        if verbose:
            print logbook.stream
            
        if checkpoint is not None and gen % freq == 0:
            os.chdir(checkpoint)
            filename = datetime.now().time().replace(microsecond=0).isoformat() + '.pkl'
            cp = dict(atlases = atlases,  generation=gen, 
                      halloffame=halloffame, logbook=logbook, 
                      migrants=migrants, constantMappings=constantMappings,
                      rndstate=random.getstate(),)
            with open(filename, 'wb') as f:
                pickle.dump(cp, f, -1)
            print "Checkpointed at generation %.i" % gen + ' in ' + checkpoint + '/' + filename
            lastcheckpoint = gen
        elif lastcheckpoint is not None:
            print "Last checkpointed at generation %.i" % lastcheckpoint + ' in ' + checkpoint + '/' + filename
        pbar.update(gen)
        print ''        
    pbar.finish()
            
    return paretos, logbook

def swap(individuals, population):
    for i in range(len(individuals)):
        population.remove(random.choice(population))
    population.extend(individuals)

def showFit(stat):
    return map(lambda x: "%.1f" % x[0], stat)

######################################
# GP Program generation functions    #
######################################

'''
These have been taken from the DEAP source code and
https://gist.github.com/macrintr/9876942 to allow more robust 
individual creation
'''

__type__ = object
        
def genFull(pset, min_, max_, type_=__type__):
    """Generate an expression where each leaf has a the same depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) no return type is enforced.
    :returns: A full tree with all leaves at the same depth.
    """
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height
    return generate(pset, min_, max_, condition, type_)

def genGrow(pset, min_, max_, type_=__type__):
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.and someone getting quite confuse

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) no return type is enforced.
    :returns: A grown tree with leaves at possibly different depths.
    """
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a a node should be a terminal.
        """
        return depth == height or \
               (depth >= min_ and random.random() < pset.terminalRatio)
    return generate(pset, min_, max_, condition, type_)

def genHalfAndHalf(pset, min_, max_, type_=__type__):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) no return type is enforced.
    :returns: Either, a full or a grown tree.
    """
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)

def genRamped(pset, min_, max_, type_=__type__):
    """
    .. deprecated:: 1.0
        The function has been renamed. Use :func:`~deap.gp.genHalfAndHalf` instead.
    """
    warnings.warn("gp.genRamped has been renamed. Use genHalfAndHalf instead.",
                  FutureWarning)
    return genHalfAndHalf(pset, min_, max_, type_)

def generate(pset, min_, max_, condition, type_=__type__):
    """Generate a Tree as a list of list. The tree is build
    from the root to the leaves, and it stop growing when the
    condition is fulfilled.
 
    :param pset: A primitive set from wich to select primitives of the trees.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) no return type is enforced.
    :returns: A grown tree with leaves at possibly different depths
              dependending on the condition function.
              
    
    DUMMY NODE ISSUES
    
    DEAP will only place terminals if we're at the bottom of a branch. 
    This creates two issues:
    1. A primitive that takes other primitives as inputs could be placed at the 
        second to last layer.
        SOLUTION: You need to allow the tree to end whenever the height condition is met,
                    so create "dummy" terminals for every type possible in the tree.
    2. A primitive that takes terminals as inputs could be placed above the second to 
        last layer.
        SOLUTION: You need to allow the tree to continue extending the branch until the
                    height condition is met, so create "dummy" primitives that just pass 
                    through the terminal types.
                    
    These "dummy" terminals and "dummy" primitives introduce unnecessary and sometimes 
    nonsensical solutions into populations. These "dummy" nodes can be eliminated
    if the height requirement is relaxed.    
    
    
    HOW TO PREVENT DUMMY NODE ISSUES
    
    Relaxing the height requirement:
    When at the bottom of the branch, check for terminals first, then primitives.
        When checking for primitives, skirt the height requirement by adjusting 
        the branch depth to be the second to last layer of the tree.
        If neither a terminal or primitive fits this node, then throw an error.
    When not at the bottom of the branch, check for primitives first, then terminals.
    
    Issue with relaxing the height requirement:
    1. Endless loops are possible when pint(x)rimitive sets have any type loops. 
        A primitive with an output of one type may not take an input type of 
        itself or a parent type.
        SOLUTION: A primitive set must be well-designed to prevent those type loops.
    
    """
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        # At the bottom of the tree
        if condition(height, depth):
            # Try finding a terminal
            try:
                term = random.choice(pset.terminals[type_])
                
                if isclass(term):
                    term = term()
                expr.append(term)                
            # No terminal fits
            except:
                # So pull the depth back one layer, and start looking for primitives
                try:
                    depth -= 1
                    prim = random.choice(pset.primitives[type_])
          
                    expr.append(prim)
                    for arg in reversed(prim.args):
                        stack.append((depth+1, arg)) 
                                    
                # No primitive fits, either - that's an error
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError, "The gp.generate function tried to add "\
                                      "a primitive of type '%s', but there is "\
                                      "none available." % (type_,), traceback
 
        # Not at the bottom of the tree
        else:
            # Check for primitives
            try:
                prim = random.choice(pset.primitives[type_])
          
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth+1, arg))                 
            # No primitive fits
            except:                
                # So check for terminals
                try:
                    term = random.choice(pset.terminals[type_])
                
                # No terminal fits, either - that's an error
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError, "The gp.generate function tried to add "\
                                      "a terminal of type '%s', but there is "\
                                      "none available." % (type_,), traceback
                if isclass(term):
                    term = term()
                expr.append(term)
 
    return expr

def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.
    
    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: A dictionary of mutation functions and their probabilities
    :returns: A list of varied individuals that are independent of their
              parents.
    
    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.
    A first loop over :math:`P_\mathrm{o}` is executed to mate consecutive
    individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting
    :math:`P_\mathrm{o}` is returned.
    
    This variation is named *And* beceause of its propention to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematicaly, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]
    
    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
            del offspring[i-1].fitness.values, offspring[i].fitness.values
            offspring[i-1].charac = None
    
    for mutator, probability in mutpb.iteritems():
        for i in range(len(offspring)):
            if random.random() < probability:
                offspring[i], = mutator(offspring[i])
                del offspring[i].fitness.values
                offspring[i].charac = None
    
    return offspring

class ConstantMaps(tools.HallOfFame):
    """The Pareto front hall of fame contains all the non-dominated individuals
    that ever lived in the population. That means that the Pareto front hall of
    fame can contain an infinity of different individuals.
    
    :param similar: A function that tels the Pareto front whether or not two
                    individuals are similar, optional.
    
    The size of the front may become very large if it is used for example on
    a continuous function with a continuous domain. In order to limit the number
    of individuals, it is possible to specify a similarity function that will
    return :data:`True` if the genotype of two individuals are similar. In that
    case only one of the two individuals will be added to the hall of fame. By
    default the similarity function is :func:`operator.eq`.
    
    Since, the Pareto front hall of fame inherits from the :class:`HallOfFame`, 
    it is sorted lexicographically at every moment.
    """
    def __init__(self, similar=operator.eq):
        tools.HallOfFame.__init__(self, None, similar)
    
    def update(self, population):
        """Update the Pareto front hall of fame with the *population* by adding 
        the individuals from the population that are not dominated by the hall
        of fame. If any individual in the hall of fame is dominated it is
        removed.
        
        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        ind_added = []
        
        for ind in population:
            is_dominated = False
            has_twin = False
            to_remove = []
            for i, hofer in enumerate(self):    # hofer = hall of famer
                if hofer.fitness.dominates(ind.fitness):
                    is_dominated = True
                    break
                elif ind.fitness.dominates(hofer.fitness):
                    to_remove.append(i)
                elif ind.fitness == hofer.fitness and self.similar(ind, hofer):
                    has_twin = True
                    break
            
            for i in reversed(to_remove):       # Remove the dominated hofer
                self.remove(i)
            if not is_dominated and not has_twin:
                self.insert(ind)
                ind_added.append(ind)
        
        return ind_added
    
def returnParetoFront(population, similar=operator.eq):
    pareto = population[:]
    for ind in population:
        is_dominated = False
        has_twin = False
        to_remove = []
        for i, hofer in enumerate(pareto):    # hofer = hall of famer
            if hofer.fitness.dominates(ind.fitness):
                is_dominated = True
                break
            elif ind.fitness.dominates(hofer.fitness):
                to_remove.append(hofer)
            elif ind.fitness == hofer.fitness and np.array_equal(ind, hofer):
                has_twin = True
                break
        for hofer in reversed(to_remove):       # Remove the dominated hofer
            pareto.remove(hofer)
        if not is_dominated and not has_twin:
            pareto.insert(ind)
    return pareto
