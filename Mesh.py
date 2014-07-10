'''
Created on 8 Jul 2014

@author: mg542
'''
import os
import numpy as np
import cPickle

import matplotlib.pyplot as plt

from CellModeller.CellState import CellState

from Description import *

directory = '/scratch/mg542/Data/CellModeller'
pickleFile = '/scratch/mg542/Data/CellModeller/1/20140704-204113/step-00550.pickle'

def plotHist(state, gridsize):
    positions = state[0]['position']
    Xpos = positions[:,0]
    Ypos = positions[:,1]
    maxX, minX = np.max(Xpos), np.min(Xpos)
    maxY, minY = np.max(Ypos), np.min(Ypos)
    
    #gridsize = 5
    nX = int((maxX-minX)/gridsize) + 3
    nY = int((maxY-minY)/gridsize) + 3
    center = (maxX+minX)/2., (maxY+minY)/2.
    extentX = center[0] - nX*gridsize/2., center[0] + nX*gridsize/2.
    extentY = center[1] - nY*gridsize/2., center[1] + nY*gridsize/2.
    Xrange = np.arange(*extentX + (gridsize,))
    Yrange = np.arange(*extentY + (gridsize,))
    
    hist, xedge, yedge = np.histogram2d(Xpos, Ypos, (Xrange, Yrange))
    plt.imshow(hist)
    plt.show()
    
    return hist, xedge, yedge

def plotHistType(state, gridsize, type=None, both=False):
    if type is not None:
        positions = state[0]['position'][state[0]['type']==type]
    elif both:
        positions = state[0]['position'][state[0]['type']==0]
    else:
        positions = state[0]['position']
    Xpos = positions[:,0]
    Ypos = positions[:,1]
    maxX, minX = np.max(Xpos), np.min(Xpos)
    maxY, minY = np.max(Ypos), np.min(Ypos)
    
    #gridsize = 5
    nX = int((maxX-minX)/gridsize) + 3
    nY = int((maxY-minY)/gridsize) + 3
    center = (maxX+minX)/2., (maxY+minY)/2.
    extentX = center[0] - nX*gridsize/2., center[0] + nX*gridsize/2.
    extentY = center[1] - nY*gridsize/2., center[1] + nY*gridsize/2.
    Xrange = np.arange(*extentX + (gridsize,))
    Yrange = np.arange(*extentY + (gridsize,))
    
    hist, xedge, yedge = np.histogram2d(Xpos, Ypos, (Xrange, Yrange))
    plt.imshow(hist)
    
    if both:
        plt.figure()
        plotHistType(state, gridsize, type=1)
    else: 
        plt.show()
    
    
    return hist, xedge, yedge

def plotHist1D(hist, bins = 20):
    freq, bins = np.histogram(hist, bins)
    plt.plot(bins[:-1], freq)
    plt.show()
    
def generateField(state, gridsize=5, type=None):
    Xrange, Yrange = findBins(state, gridsize)
    positions = state[0]['position']
    if type is not None:
        positions = state[0]['position'][state[0]['type']==type]
        
    Xpos = positions[:,0]
    Ypos = positions[:,1]    
    
    hist, xedge, yedge = np.histogram2d(Xpos, Ypos, (Xrange, Yrange))
    
    return hist, xedge, yedge

def plotLines(state, gridsize=5, type=None):
    h,x,y = generateField(state, gridsize, type)
    for line in h:
        plt.plot(line)
    plt.show()

def findMesh(state, gridsize):
    positions = state[0]['position']
    Xpos = positions[:,0]
    Ypos = positions[:,1]
    maxX, minX = np.max(Xpos), np.min(Xpos)
    maxY, minY = np.max(Ypos), np.min(Ypos)
    
    gridsize = float(gridsize)
    nX = int((maxX-minX)/gridsize) + 3
    nY = int((maxY-minY)/gridsize) + 3
    center = (maxX+minX)/2., (maxY+minY)/2.
    extentX = center[0] - nX*gridsize/2., center[0] + nX*gridsize/2.
    extentY = center[1] - nY*gridsize/2., center[1] + nY*gridsize/2.
    Xrange = np.arange(*extentX + (gridsize,))
    Yrange = np.arange(*extentY + (gridsize,))
    return extentX + extentY + (gridsize,)

def findBins(state,gridsize):
    minX, maxX, minY, maxY, gridsize = findMesh(state, gridsize)
    Xrange = np.arange(minX, maxX, gridsize)
    Yrange = np.arange(minY, maxY, gridsize)
    return Xrange, Yrange

def plotField(field):
    plt.imshow(field)
    plt.show()
    
def plotIntersection(state, gridsize):
    cell0 = generateField(state,gridsize, 0)[0]
    cell1 = generateField(state,gridsize, 1)[0]
    print (cell0 * cell1).sum()
    plotField(cell0 * cell1)
    return cell0 * cell1

def plotContours(contours):
    for i, contour in enumerate(contours):
        plt.plot(contour[:,0,0], contour[:,0,1])
        
def threshold(image):
    im = np.array(image, dtype=np.uint8)
    ret, thresh = cv2.threshold(im,1,256,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

if __name__ == '__main__':
    state = State(file=pickleFile)
    gridsize = 7
    mesh = findMesh(state, gridsize)
    cell0 = generateField(state,gridsize, 0)[0]
    cell1 = generateField(state,gridsize, 1)[0]
    plotField(cell0 * cell1)
    
    import cv2
    im = np.array(cell0 * cell1, dtype=np.uint8)
    ret, thresh = cv2.threshold(im,1,256,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    