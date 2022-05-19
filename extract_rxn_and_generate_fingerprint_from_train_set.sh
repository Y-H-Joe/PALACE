#!/bin/bash

##SBATCH --partition=general
#ESBATCH --partition=bigmem4
#SBATCH --partition=bigmem2
##SBATCH --partition=gpu4
##SBATCH --partition=gpu2
##SBATCH --partition=amd
## sinfo

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=90-00:00:00
#SBATCH --mem=300G
##SBATCH --exclusive
#SBATCH --ntasks-per-node=1
##SBATCH --gres=gpu:tesla:2
##SBATCH --job-name "yihang's job"

#SBATCH --chdir /scratch/yzz0191/metabolism_20220127/
#SBATCH --error /scratch/yzz0191/metabolism_20220127/code/e/e.extract.%A_%a.txt  ######
#SBATCH --output /scratch/yzz0191/metabolism_20220127/code/e/o.extract.%A_%a.txt  ######
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

/home/yzz0191/anaconda3/envs/PALACE/bin/python3 extract_rxn_and_generate_fingerprint_from_train_set.py




