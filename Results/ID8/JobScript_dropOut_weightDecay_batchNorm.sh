#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
##BSUB -q gpuk80
### -- set the job Name --
#BSUB -J ResNet152_dropOut_weightDecay_batchNorm_300epochs
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 10GB of memory
#BSUB -R "rusage[mem=10GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start  --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

#nvidia-smi

#Load cuda, pandas and python
module load cuda/9.1
module load python3/3.6.2
module load pandas/0.20.3-python-3.6.2
#Install relevant packages
#pip3 install --user http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
#pip3 install --user torchvision
#pip3 install --user pytorch-ignite
#pip3 install --user scikit-image
#pip3 install --user scikit-learn
#pip3 install --user Pillow
#For pytorch text
#pip3 install --user torchtext

echo "Executing COFGA_resnet152_dropOut_weightDecay_batchNorm_300epochs"
python3 ./COFGA_resnet152_dropOut_weightDecay_batchNorm.py
