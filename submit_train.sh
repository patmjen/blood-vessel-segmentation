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
#BSUB -W 16:00
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

python run_training.py --experiment_name=vnet_train_avizo_long --max_epochs=5000 --num_loader_workers=0 --logger_save_dir=/work1/patmjen/logs/november --samples_per_volume=24 --batch_size=2
