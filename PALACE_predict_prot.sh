#!/bin/bash

##SBATCH --partition=general
##SBATCH --partition=bigmem4
##SBATCH --partition=bigmem2
##SBATCH --partition=gpu4
#SBATCH --partition=gpu2
## sinfo

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=90-00:00:00
##SBATCH --mem=5G
##SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:tesla:1
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

version=EC_task3_notransfer
smi_vocab_dp=vocab/smi_vocab_v2.pkl
prot_vocab_dp=vocab/prot_vocab.pkl
class_num=5152

#for piece in {1..14} # EC_task1 class_num=2
#for piece in {0..3} # EC_task2 class_num=8
#for piece in {0..48} # EC_task3 class_num=5152
for piece in {0..50} # EC_task3_notransfer class_num=5152
do
     if test ! -f PALACE_predictions_EC/PALACE_${version}_piece_${piece}_prediction.txt;then
        ~/anaconda3/envs/PALACE/bin/python PALACE_predict_prot.py PALACE_models/PALACE_${version}_piece_${piece}.pt data/PALACE_EC_task1.test.tsv $smi_vocab_dp $prot_vocab_dp PALACE_predictions_EC/PALACE_${version}_piece_${piece}_prediction.txt ${class_num}
     fi
done

