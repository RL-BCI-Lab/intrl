#!/bin/bash

# Algo Naming Scheme:
######################################################################################
# bcn = behavioral cloning noise
# <number>p = Number of ensemble policies used (i.e., networks trained)
# nosplit = Data is not partitioned based on number of policies. 

# Global args
######################################################################################
wandb=offline
run_all=false

# hydra_job='--cfg job'
hydra_job='-m'
# hydra_job='--hydra-help'

# Train
######################################################################################
exp_group=tests
data_name=backward-70%
demo_paths=['$\{hydra:runtime.cwd\}/demos/tests/$\{env_spec.vars.name\}/human/backward-70%/rid0/collect']
demo_file_pattern='traj.*\.h5'

# BCNoise with 1 policy (default 500 epochs, 5 iterations)
###################################################################################
if false || ${run_all} ; then
train=true # Enable training
eval=true # Enable eval
algo_name='bcn-policies\=1'

if ${train} || ${run_all} ; then
    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='imitate' \
    +experiment=mountain_car/bc/train \
    imitation/imitator=bcn-disc \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=True \
    ++imitation.vars.demo_file_pattern=${demo_file_pattern} \
    ++imitation.vars.demo_paths=${demo_paths}
fi

if ${eval} || ${run_all} ; then
    run_id=rid0

    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='eval' \
    +experiment=mountain_car/bc/eval \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++output_dir.run_id=${run_id} \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=True 
fi
fi

# BCNoise with 5 policy ensemble (default 500 epochs, 5 iterations)
###################################################################################
if false || ${run_all} ; then
train=true # Enable training
eval=true # Enable eval
algo_name='bcn-policies\=5'

if ${train} || ${run_all} ; then
    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='imitate' \
    +experiment=mountain_car/bc/train \
    imitation/imitator=bcn-disc-5p \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=True \
    ++imitation.vars.demo_file_pattern=${demo_file_pattern} \
    ++imitation.vars.demo_paths=${demo_paths}
fi

if ${eval} || ${run_all} ; then
    run_id=rid0

    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='eval' \
    +experiment=mountain_car/bc/eval \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++output_dir.run_id=${run_id} \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=True 
fi
fi

# Replication 1 policy ensemble
######################################################################################
if false || ${run_all} ; then
train=false # Enable training
eval=false # Enable eval
algo_name='bcn-policies\=1'
data_name=backward-70%

if ${train} || ${run_all} ; then
    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='imitate' \
    +experiment=mountain_car/bc/train \
    imitation/imitator=bcn-disc \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++output_dir.run_id='cpu-replication' \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=True \
    ++imitation.imitator.instance.policy_specs.0.kwargs.mlp_extractor_kwargs.device=cpu \
    ++imitation.imitator.instance.rng.seed=0 \
    ++imitation.vars.demo_file_pattern=${demo_file_pattern} \
    ++imitation.vars.demo_paths=${demo_paths}
fi

if ${eval} || ${run_all} ; then
    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='eval' \
    +experiment=mountain_car/bc/eval \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++output_dir.run_id='cpu-replication' \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=True \
    ++evaluation.evaluator.func.env_seed=0
fi
fi

#  Replication 5 policy ensemble
######################################################################################
if false || ${run_all} ; then
train=true # Enable training
eval=true # Enable eval
algo_name='bcn-policies\=5'
data_name=backward-70%

if ${train} || ${run_all} ; then
    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='imitate' \
    +experiment=mountain_car/bc/train \
    imitation/imitator=bcn-disc-5p \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++output_dir.run_id='cpu-replication' \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=True \
    ++imitation.imitator.instance.policy_specs.0.kwargs.mlp_extractor_kwargs.device=cpu \
    ++imitation.imitator.instance.rng.seed=0 \
    ++imitation.vars.demo_file_pattern=${demo_file_pattern} \
    ++imitation.vars.demo_paths=${demo_paths}
fi

if ${eval} || ${run_all} ; then
    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='eval' \
    +experiment=mountain_car/bc/eval \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++output_dir.run_id='cpu-replication' \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=True \
    ++evaluation.evaluator.func.env_seed=0
fi
fi
