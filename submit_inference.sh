#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J infer
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 4:00
# specify system resources
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
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
#BSUB -o batch_output/inference_%J.out
#BSUB -e batch_output/inference_%J.err
# -- end of LSF options --

# Load the cuda module
source init.sh

nvidia-smi

git log -1 --no-color
git --no-pager diff -U1

# JOB_IDS="9093176" # Best covid-19
JOB_IDS="8904512" # Best control
# JOB_IDS="9204292" # Best big covid-19
# JOB_IDS="9325782" # Best corrected big covid-19

LOG_DIR=/work1/patmjen/HALOS/logs

for jid in $JOB_IDS; do
	echo $jid
	CKPT_DIR=$LOG_DIR/net_train_$jid/ckpts
	CKPT=$CKPT_DIR/$(ls -1 $CKPT_DIR)
        python run_inference.py $CKPT \
		--model=vnet \
		--crop_size=128 \
		--data_dir=/work1/patmjen/HALOS/inference/data/control \
		--save_dir=/work1/patmjen/HALOS/inference/results/control
done

