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

