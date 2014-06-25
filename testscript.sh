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


# Path to your executable. For example, if you extracted SCOOP to $HOME/downloads/scoop
cd $HOME/Source/HOTBOT

# Add any addition to your environment variables like PATH. For example, if your local python installation is in $HOME/python
#export PATH=$HOME/python/bin:$PATH

# If, instead, you are using the python offered by the system, you can stipulate it's library path via PYTHONPATH
#export PYTHONPATH=$HOME/wanted/path/lib/python+version/site-packages/:$PYTHONPATH
# Or use VirtualEnv via virtualenvwrapper here:
#workon yourenvironment

# Launch SCOOP using the hosts
python -m scoop -vv full_tree.py > output

echo
echo "Job finished. PBS details are:"
echo
qstat -f ${PBS_JOBID}
echo
