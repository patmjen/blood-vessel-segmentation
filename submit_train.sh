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
#BSUB -R "rusage[mem=12GB]"
##BSUB -R "select[gpu32gb]"
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
#BSUB -o train_%J.out
#BSUB -e train_%J.err
# -- end of LSF options --

# Load the cuda module
source init.sh

nvidia-smi

git log -1 --no-color
git --no-pager diff -U1

python run_training.py \
    unet \
    --experiment_name=vnet_train_${LSB_JOBID} \
    --lr=1e-3 \
    --max_epochs=10000 \
    --num_loader_workers=0 \
    --logger_save_dir=/work1/patmjen/logs/december/ \
    --data_dir=/work1/patmjen/data/sparse3d/ \
    --samples_per_volume=256 \
    --batch_size=2 \
    --progress_bar_refresh_rate=0 \
    --crop_size=96 \

