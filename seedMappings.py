'''
Created on 15 Jul 2014

@author: mg542
'''

import numpy as np
from deap import gp
from storePrimitives import *
from Primitives import *

psetFile = '/home/mg542/Documents/Source/HOTBOT/pset.pkl'
domainFile = '/scratch/mg542/CellModeller/domain.pkl'
seedFile = '/home/mg542/Documents/Source/HOTBOT/seed.pkl'

['safeDiv_float-NumberField',
 'multiply_CellStates-CellStates',
 'add_float-CellStates',
 'safePower_CellStates-CellStates',
 'sumPerimeters_Contours',
 'lVectorX_CellPositions',
 'vectorY_Vector',
 'lVectorX_CellVectors',
 'log_Density',
 'cross_Vector-Position',
 'sumVector_CellPositions',
 'cross_CellPositions-CellPositions',
 'sqrt_Density',
 'less_NumberField-float',
 'negative_CellStates',
 'sumVector_CellVectors',
 'dot_Vector-Position',
 'exp_float',
 'log_float',
 'Density_CellPositions-Mesh-CellTypes-CellType',
 'nematicOrder_CellPositions-CellVectors-Mesh',
 'cross_CellVectors-CellVectors',
 'solidity_Contour',
 'ARG4',
 'add_CellStates-float',
 'add_float-Density',
 'densityMultiply_NumberField-NumberField',
 'perimeter_Contour',
 'negative_Density',
 'cross_Vector-CellVectors',
 'subtract_float-Density',
 'False',
 'cross_CellPositions-CellVectors',
 'largestContour_Contours',
 'greater_NumberField-float',
 'cross_CellVectors-Position',
 'safeDiv_CellStates-float',
 'NumberField_CellPositions-CellStates-Mesh',
 'less_CellStates-CellStates',
 'safePower_float-Density',
 'greater_CellStates-float',
 'safePower_float-NumberField',
 'multiply_Vector-Vector',
 'log_CellStates',
 'centroidY_Contour',
 'lVectorZ_CellPositions',
 'subtract_float-CellStates',
 'less_float-CellStates',
 'vectorZ_Vector',
 'convexHull_Contour',
 'nematicOrder_CellPositions-CellVectors-Mesh-CellTypes-CellType',
 'greater_CellStates-CellStates',
 'multiply_Position-CellVectors',
 'greater_float-CellStates',
 'lVectorY_CellVectors',
 'negative_NumberField',
 'multiply_CellStates-float',
 'multiply_CellPositions-CellVectors',
 'multiply_Vector-CellPositions',
 'add_float-NumberField',
 'multiply_CellPositions-Vector',
 'multiply_CellVectors-CellPositions',
 'vectorX_Position',
 'less_float-NumberField',
 'multiply_float-NumberField',
 'cross_Position-Position',
 'cross_Position-CellPositions',
 'True',
 'safeDiv_float-float',
 'absolute_Density',
 'densityMultiply_Density-NumberField',
 'multiply_CellVectors-Position',
 'sum_CellStates',
 'multiply_Vector-CellVectors',
 'multiply_Position-CellPositions',
 'subtract_CellStates-CellStates',
 'greater_float-Density',
 'greater_float-NumberField',
 'safePower_Density-float',
 'multiply_CellVectors-CellVectors',
 'densityMultiply_NumberField-Density',
 'ARG0',
 'ARG1',
 'ARG2',
 'ARG3',
 'multiply_Position-Position',
 'ARG5',
 'ARG6',
 'safePower_CellStates-float',
 'subtract_CellStates-float',
 'sqrt_NumberField',
 'cross_CellVectors-Vector',
 'adfContour_CellPositions-Mesh-CellTypes-CellType',
 'add_NumberField-float',
 'less_Density-float',
 'vectorY_Position',
 'multiply_CellPositions-CellPositions',
 'cross_CellVectors-CellPositions',
 'safeDiv_float-CellStates',
 'multiply_Density-float',
 'norm_Vector',
 'exp_CellStates',
 'absolute_float',
 'greater_Density-float',
 'exp_NumberField',
 'greater_float-float',
 'absolute_CellStates',
 'cross_CellPositions-Position',
 'multiply_float-Density',
 'exp_Density',
 'if_then_else_bool-float-float',
 'densityMultiply_Density-Density',
 'multiply_Vector-Position',
 'vectorX_Vector',
 'subtract_float-float',
 'add_Density-float',
 'norm_CellPositions',
 'multiply_CellVectors-Vector',
 'log_NumberField',
 'sqrt_CellStates',
 'Contours_NumberField-float-Mesh',
 'cross_Position-CellVectors',
 'lVectorZ_CellVectors',
 'add_CellStates-CellStates',
 'multiply_float-float',
 'centroidX_Contour',
 'vectorZ_Position',
 'safePower_NumberField-float',
 'adfContour_CellPositions-Mesh',
 'NumberField_CellPositions-CellStates-Mesh-CellTypes-CellType',
 'less_float-Density',
 'dot_Vector-Vector',
 'subtract_float-NumberField',
 'area_Contour',
 'absolute_NumberField',
 'less_float-float',
 'multiply_NumberField-float',
 'Density_CellPositions-Mesh',
 'safeDiv_CellStates-CellStates',
 'multiply_CellPositions-Position',
 'subtract_Density-float',
 'safeDiv_NumberField-float',
 'safeDiv_Density-float',
 'norm_CellVectors',
 'lVectorY_CellPositions',
 'cross_Vector-CellPositions',
 'add_float-float',
 'Contours_Density-float-Mesh',
 'cross_Position-Vector',
 'multiply_Position-Vector',
 'sqrt_float',
 'safePower_float-float',
 'multiply_float-CellStates',
 'cross_Vector-Vector',
 'negative_float',
 'safePower_float-CellStates',
 'subtract_NumberField-float',
 'safeDiv_float-Density',
 'less_CellStates-float',
 'cross_CellPositions-Vector',
 'norm_Position']

inputTypes = [float, CellTypes, CellStates, CellStates,
                  CellPositions, CellVectors, Mesh]

def strFunc(func, args):
    args = ', '.join(args)
    return func + args +')'

cntsDensStr = 'Contours_DensityfloatMesh('
cntsNFieldStr = 'Contours_NumberFieldfloatMesh('
thresh = '1.'
mesh = 'mesh'

densdensMulStr = 'densityMultiply_DensityDensity('
densNfieldMulStr = 'densityMultiply_DensityNumberField('
NFieldNfieldMulStr = 'densityMultiply_NumberFieldNumberField('

largestContourStr = 'largestContour_Contours('

contourFStrs = ['solidity_Contour(', 
                'perimeter_Contour(',
                'centroidY_Contour(',
                'centroidX_Contour(',
                'orientation_Contour(',
                'majorAxis_Contour(',
                'minorAxis_Contour(',
                'area_Contour(',
                ]

ContoursStrs = ['sumPerimeters_Contours(',
                'sumArea_Contours(',
                ]

densityStrs = ['Density_CellPositionsMesh(positions, mesh)',
               'Density_CellPositionsMeshCellTypesCellType(positions, mesh, types, CellType0)',
               'Density_CellPositionsMeshCellTypesCellType(positions, mesh, types, CellType1)',]

nfieldStrs = ['nematicOrder_CellPositionsCellVectorsMesh(positions, directions, mesh)', 
              'nematicOrder_CellPositionsCellVectorsMeshCellTypesCellType(positions, directions, mesh, types, CellType0)', 
              'nematicOrder_CellPositionsCellVectorsMeshCellTypesCellType(positions, directions, mesh, types, CellType1)', 
              'NumberField_CellPositionsCellStatesMesh(positions, lengths, mesh)',
              ]

newdensstrs1 = [strFunc(densdensMulStr, (dens1, dens2)) for dens1 in densityStrs for dens2 in densityStrs]
newdensstrs2 = [strFunc(densNfieldMulStr, (dens1, nfield2)) for dens1 in densityStrs for nfield2 in nfieldStrs]
densityStrs.extend(newdensstrs1 + newdensstrs2)
nfieldStrs.extend([strFunc(NFieldNfieldMulStr, (nfield1, nfield2)) for nfield1 in nfieldStrs for nfield2 in nfieldStrs])

densityContours = [strFunc(cntsDensStr, [dens, thresh, mesh]) for dens in densityStrs]
nfieldContours = [strFunc(cntsNFieldStr, [nfield, thresh, mesh]) for nfield in nfieldStrs]

contourStrs = [strFunc(largestContourStr, [cntsStr]) for cntsStr in densityContours + nfieldContours]

mapStrs = [strFunc(cntFunc, [cntStr]) for cntFunc in contourFStrs for cntStr in contourStrs]

fullDensity = 'Density_CellPositionsMesh(positions, mesh)'
density0 = 'Density_CellPositionsMeshCellTypesCellType(positions, mesh, types, CellType0)'
density1 = 'Density_CellPositionsMeshCellTypesCellType(positions, mesh, types, CellType2)'
densityMul = 'densityMultiply_DensityDensity('
solidityStr = 'solidity_Contour('

combinedDensity = strFunc(densityMul, [ density0, density1])

solFuncStr = 'solidity_Contour(largestContour_Contours(Contours_DensityfloatMesh(densityMultiply_DensityDensity(Density_CellPositionsMeshCellTypesCellType(positions, mesh, types, CellType0), Density_CellPositionsMeshCellTypesCellType(positions, mesh, types, CellType1)),1., mesh)))'
'solidity_Contour(largestContour_Contours(Contours_DensityfloatMesh(Density_CellPositionsMeshCellTypesCellType(positions, mesh, types, CellType0), 1., mesh)))'
'solidity_Contour(adfContour_CellPositionsMesh(ARG4, ARG6))'

goodMappings = ['solidity_Contour(largestContour_Contours(Contours_DensityfloatMesh']

def plotCharac(charac):
    import matplotlib.pyplot as plt
    values = charac['characteristic']
    for sim in values:
        plt.figure()
        for run in sim:
            plt.plot(run)
    plt.show()


if __name__ == '__main__':
    pset = loadPset(psetFile)
    with open(domainFile, 'rb') as f:
        domain = cPickle.load(f)
    state = domain[0][0].states[0]
    primtree = gp.PrimitiveTree([])
    ind = primtree.from_string(solFuncStr, pset)
    characteristic = domain.evalCharacteristics(gp.compile(ind, pset))
    seedMaps = [primtree.from_string(str, pset) for str in mapStrs]
    for str in mapStrs:
        primtree.from_string(str, pset)
        
    with open(seedFile, 'wb') as f:
        cPickle.dump(seedMaps, f, -1)
    
    
