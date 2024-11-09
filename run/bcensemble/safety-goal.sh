#!/bin/bash

# Algo Naming Scheme:
######################################################################################
# bc = behavioral cloning
# bce = behavioral cloning ensemble
# polices = Number of ensemble policies used (i.e., networks trained)
# nosplit = Data is not partitioned based on number of policies. 

# Global args
######################################################################################
wandb=offline
run_all=false

# hydra_job='--cfg job'
hydra_job='-m'
# hydra_job='--hydra-help'

# Train & Eval
######################################################################################
exp_group=tests
data_name=avoid-area
demos=['$\{hydra:runtime.cwd\}/demos/tests/$\{env_spec.vars.name\}/human/avoid-area/rid0/collect']

######################################################################################
if true || ${run_all} ; then
train=true # Enable training
eval=false # Enable eval
algo_name=bc

# Training using BC for multi-discrete action space
if ${train} || ${run_all} ; then
    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='imitate' \
    +experiment=mujoco/safety/bc/train-remap-actions \
    imitation/imitator=bc-multidisc \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=True \
    ++imitation.imitator.instance.demonstrations.trajectories.trajectories.paths=${demos}
fi

# Evaluation of BCEnsemble policy with multi-discrete action space
# Need to specify experiment, exp_group, data_name, and run_id these form a path
# Given the path, the default behavior is to load the existing policy from the 
# train/ directory. This behavior can be modified.
if ${eval} || ${run_all} ; then
    run_id=rid0

    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='eval' \
    +experiment=mujoco/safety/bc/eval-remap-actions \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++output_dir.run_id=${run_id} \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=True 
    fi

fi

######################################################################################
if false || ${run_all} ; then
train=true # Enable training
eval=true # Enable eval
algo_name=bc-cont

# Training using BC for continuous action space
if ${train} || ${run_all} ; then
    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='imitate' \
    +experiment=mujoco/safety/bc/train-cont \
    imitation/imitator/instance=bc-cont \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=True \
    ++imitation.imitator.instance.demonstrations.trajectories.paths=${demos}
fi

if ${eval} || ${run_all} ; then
    run_id=rid0

    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='eval' \
    +experiment=mujoco/safety/bc/eval \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++output_dir.run_id=${run_id} \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=True 
    fi

fi




