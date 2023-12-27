#!/bin/sh

## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=ann-closed-roa-grid-runner17

## General partitions: all-HiPri, bigmem-HiPri   --   (12 hour limit)
##                     all-LoPri, bigmem-LoPri, gpuq  (5 days limit)
## Restricted: CDS_q, CS_q, STATS_q, HH_q, GA_q, ES_q, COS_q  (10 day limit)
#SBATCH --partition=gpuq

## Parameters for using GPU partition
## Step 1 - Specify the number of GPUS
## Step 2 - Specity the quality of service
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --qos=gpu

## Separate output and error messages into 2 files.`
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID`
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file`
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file`

## Slurm can send you updates via email
#SBATCH --mail-type=ALL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=ssnyde9@gmu.edu     # Put your GMU email address here

## Specify how much memory your job needs. (2G is the default)
#SBATCH --mem=32G        # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=0-23:59  # Total time needed for job: Days-Hours:Minutes

## Run your program or script
source /home/${USER}/.bashrc
cd /home/${USER}/dev/Deep_ROA_BO/
conda activate deep-roa-bo-ann
python ./ann/closed_roa/runners/grid_runner17.py