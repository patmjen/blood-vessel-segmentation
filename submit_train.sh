#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J train
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# specify system resources
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o batch_output/train_%J.out
#BSUB -e batch_output/train_%J.err
# -- end of LSF options --

# Load the cuda module
source init.sh

nvidia-smi

git log -1 --no-color
git --no-pager diff -U1

python run_training.py \
    --experiment_name=net_train_${LSB_JOBID} \
    --logger_save_dir=/work1/patmjen/HALOS/logs/ \
    --max_epochs=10000 \
    --progress_bar_refresh_rate=0 \
    vnet \
    --lr=1e-3 \
    --num_loader_workers=0 \
    --data_dir=/work1/patmjen/HALOS/data/big_sparse/ \
    --samples_per_volume=256 \
    --batch_size=12 \
    --crop_size=96 \
    --normalization='b' \
    --min_lr=1e-3 \
    --lr_reduce_factor=0.8 \

