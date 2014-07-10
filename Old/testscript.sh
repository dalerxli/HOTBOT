#!/bin/bash
## Please refer to your grid documentation for available flags. This is only an example.
#PBS -q s48
##PBS -l procs=16
#PBS -V
#PBS -N SCOOPJob



echo "Starting job $PBS_JOBID"
echo
echo "PBS assigned me this node:"
cat $PBS_NODEFILE
echo
echo "PBS_O_WORKDIR"
echo PBS_O_WORKDIR

# Path to your executable. For example, if you extracted SCOOP to $HOME/downloads/scoop
cd $HOME/Source/HOTBOT

# Add any addition to your environment variables like PATH. For example, if your local python installation is in $HOME/python
#export PATH=$HOME/python/bin:$PATH

# If, instead, you are using the python offered by the system, you can stipulate it's library path via PYTHONPATH
#export PYTHONPATH=$HOME/wanted/path/lib/python+version/site-packages/:$PYTHONPATH
# Or use VirtualEnv via virtualenvwrapper here:
#workon yourenvironment

cd $PBS_O_WORKDIR
TMP=/scratch/cjf41/$PBS_JOBID
mkdir -p $TMP
cd $TMP
python -m scoop -vv $HOME/Source/HOTBOT/full_tree.py > output
cp * $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
rm -rf $TMP

# Launch SCOOP using the hosts
python -m scoop -vv $HOME/Source/HOTBOT/full_tree.py > output

echo
echo "Job finished. PBS details are:"
echo
qstat -f ${PBS_JOBID}
echo
