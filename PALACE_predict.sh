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
#SBATCH --time=15-00:00:00
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

version=v3_again_again
smi_vocab_dp=vocab/smi_vocab_v2.pkl
prot_vocab_dp=vocab/prot_vocab.pkl

# for piece in {833..928} # v1 large
# for piece in {1..190} # v5 base
# for piece in {1..97} # v3 large
# for piece in {1..300} # v4 base
# for piece in {1..8} # v4_again base
# for piece in {27..53} # v5_again_again base
# for piece in {1..26} # v4_again_again base
# for piece in {1..26} # v4_again_again base again_again_round1
# for piece in {1..26} # v3_again_again large again_again_round1
# for piece in {27..53} # v1_again_again large again_again_round1
# for piece in {27..53} # v1_again_again large again_again_round2
for piece in {1..26} # v3_again_again large again_again_round2
do
     if test ! -f PALACE_predictions/again_again_round2/PALACE_${version}_piece_${piece}_prediction.txt;then
        ~/anaconda3/envs/PALACE/bin/python PALACE_predict_large.py PALACE_models/PALACE_${version}_piece_${piece}.pt data/PALACE_test.sample.tsv $smi_vocab_dp $prot_vocab_dp PALACE_predictions/again_again_round2/PALACE_${version}_piece_${piece}_prediction.txt
     fi
done

