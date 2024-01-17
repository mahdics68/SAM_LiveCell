#!/bin/bash
#SBATCH --job-name=train-UNETR-sam
#SBATCH -o UNETR-SAM-b-livecell-all-60-train-%J
#SBATCH -t 06:00:00                  
#SBATCH -p grete:shared              
#SBATCH -G A100:1                   
#SBATCH --mem 40

source ~/.bashrc
conda activate torch2-em
 

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"
 
# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V


python /home/nimmahen/code/sam_UNETR_all.py --train --input /scratch-grete/projects/nim00007/data/LiveCELL/ --save_root /scratch-grete/usr/nimmahen/models/UNETR/sam/checkpoints/livecell_all_60_vit_b --iterations 10000 --checkpoint /scratch-grete/usr/nimmahen/models/SAM/checkpoints/sam_vit_b_01ec64.pth --backbone "sam" --encoder "vit_b"
