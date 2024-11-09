#!/bin/bash

######################################################################################
# Naming Scheme:
######################################################################################

######################################################################################
# Global args
######################################################################################
# hydra_job='--cfg job'
hydra_job='-m'
# hydra_job='--hydra-help'

######################################################################################
# MountainCar
######################################################################################

# Test
# Requires an existing path env_name/algo_name/data_name/run_id to have a collect/
# sub-directory in order to run.
######################################################################################
if false ; then
experiment=label-with-pygame
exp_group=tests
env_name=MountainCar-v0
algo_name=human
data_name=test
run_id=rid1
fps=30
# pipeline_stage=fb-test # Used to change the feedback label key and pipeline stage dir name

HYDRA_FULL_ERROR=1 python main.py ${hydra_job} -cn=feedback \
+experiment=${experiment} \
++output_dir.exp_group=${exp_group} \
++output_dir.env_name=${env_name} \
++output_dir.algo_name=${algo_name} \
++output_dir.data_name=${data_name} \
++output_dir.run_id=${run_id} \
++replay.replayer.instance.fps=${fps}
fi

# Label Safety Gym Demos
######################################################################################
if true ; then
experiment=label-with-pygame
exp_group=tests
env_name=SafetyPointGoalBase-v0
algo_name=human
data_name=avoid-area
run_id=rid1
fps=60

HYDRA_FULL_ERROR=1 python main.py ${hydra_job} -cn=feedback \
+experiment=${experiment} \
++output_dir.exp_group=${exp_group} \
++output_dir.env_name=${env_name} \
++output_dir.algo_name=${algo_name} \
++output_dir.data_name=${data_name} \
++output_dir.run_id=${run_id} \
++replay.replayer.instance.fps=${fps}
fi