
import cPickle
import os
from CellModeller.CellState import CellState
from CellModeller import Simulator

pickleDir = "/home/mg542/scratch/biophysCL-1404488584"

"""
Cell state attributes:
state.id
state.idx
state.pos = [self.cell_centers[i][j] for j in range(3)]
state.dir = [self.cell_dirs[i][j] for j in range(3)]
state.radius = self.cell_rads[i]
state.length = self.cell_lens[i]
"""

if __name__ == '__main__':

    pickles = [file for file in os.listdir(pickleDir) if 'pickle' in file]

    os.chdir(pickleDir)

    def importData(pickleFile):
        with open(pickleFile, 'rb') as f:
            data = cPickle.load(f)
        return data

    simData = map(importData, pickles)

