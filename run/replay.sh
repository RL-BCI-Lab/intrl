#!/bin/bash

# Naming Scheme:
######################################################################################

# Global args
######################################################################################
# hydra_job='--cfg job'
hydra_job='-m'
# hydra_job='--hydra-help'

# Replay Demos
# Replay demos (easiest way is to replicate the existing path using env_name/algo_name/run_id)
######################################################################################
if false ; then
experiment=replay-with-pygame
exp_group=tests
env_name=MountainCar-v0
algo_name=human
data_name=test
run_id=rid0

HYDRA_FULL_ERROR=1 python main.py ${hydra_job} -cn=replay \
+experiment=${experiment} \
++output_dir.exp_group=${exp_group} \
++output_dir.env_name=${env_name} \
++output_dir.algo_name=${algo_name} \
++output_dir.data_name=${data_name} \
++output_dir.run_id=${run_id} 
fi

# Replay Safety Gym Demos
######################################################################################
if false ; then
experiment=replay-with-pygame
exp_group=tests
env_name=SafetyPointGoalBase-v0
algo_name=human
data_name=avoid-area
run_id=rid0

HYDRA_FULL_ERROR=1 python main.py ${hydra_job} -cn=replay \
+experiment=${experiment} \
++output_dir.exp_group=${exp_group} \
++output_dir.env_name=${env_name} \
++output_dir.algo_name=${algo_name} \
++output_dir.data_name=${data_name} \
++output_dir.run_id=${run_id} 
fi