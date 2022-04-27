#!/bin/bash

##SBATCH --partition=general
##SBATCH --partition=bigmem4
##SBATCH --partition=bigmem2
##SBATCH --partition=gpu4
#SBATCH --partition=gpu2
## sinfo

#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=90-00:00:00
##SBATCH --mem=5G
#SBATCH --exclusive
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:tesla:2
##SBATCH --job-name "yihang's job"

#SBATCH --chdir /scratch/yzz0191/metabolism_20220127/
#SBATCH --error /scratch/yzz0191/metabolism_20220127/code/e/e.run.%A_%a.txt  ######
#SBATCH --output /scratch/yzz0191/metabolism_20220127/code/e/o.run.%A_%a.txt  ######
##SBATCH --array 2-6

#SBATCH --mail-type=ALL
#SBATCH --mail-user=yzz0191@auburn.edu

#########################################################################
###########!!!!!!!!!!!!!    Remember to change SBATCH --array    !!!!!!!!!!!!!!!!################
#########################################################################
dp=/scratch/yzz0191/metabolism_20220127/PALACE   #####
cd ${dp}

:<<"Cm"
taskID=$SLURM_ARRAY_TASK_ID
NAMEFILE=code/namefile.cdhit.txt
FqNAME=`sed -n "${taskID}p" $NAMEFILE`
Cm

module load cuda11.0/toolkit
world_size=2

for piece in {0..14}
do
if test ! -f PALACE_models/PALACE_EC_task3_piece_${piece}.pt;then
    ~/anaconda3/envs/PALACE/bin/python PALACE_EC_task3_train.py $piece $world_size
fi
done

# ~/anaconda3/envs/PALACE/bin/python PALACE_v2_train.py 0 $world_size
