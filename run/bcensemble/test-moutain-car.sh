#!/bin/bash

# Algo Naming Scheme:
######################################################################################
# bc = behavioral cloning
# bce = behavioral cloning ensemble
# <number>p = Number of ensemble policies used (i.e., networks trained)
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
data_name=backward-70%
demo_paths=['$\{hydra:runtime.cwd\}/demos/tests/$\{env_spec.vars.name\}/human/backward-70%/rid0/collect']
demo_file_pattern='traj.*\.h5'

######################################################################################
if false || ${run_all} ; then
train=true # Enable training
eval=true # Enable eval
algo_name=bc

if ${train} || ${run_all} ; then
    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='imitate' \
    +experiment=mountain_car/bc/train \
    imitation/imitator=bc-disc \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=False \
    ++imitation.vars.demo_file_pattern=${demo_file_pattern} \
    ++imitation.vars.demo_paths=${demo_paths}
fi

if ${eval} || ${run_all} ; then
    run_id=rid0 # Update to match rid generated by train

    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='eval' \
    +experiment=mountain_car/bc/eval \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++output_dir.run_id=${run_id} \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=False 
fi
fi

######################################################################################
if false || ${run_all} ; then
train=true # Enable training
eval=true # Enable eval
algo_name="bce-polices\=2"

if ${train} || ${run_all} ; then
HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='imitate' \
    +experiment=mountain_car/bc/train \
    imitation/imitator=bce-disc-2p \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=True \
    ++imitation.vars.demo_file_pattern=${demo_file_pattern} \
    ++imitation.vars.demo_paths=${demo_paths}
fi

if ${eval} || ${run_all} ; then
    run_id=rid0 # Update to match rid generated by train

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

######################################################################################
if false || ${run_all} ; then
train=true # Enable training
eval=true # Enable eval
algo_name='bce-polices\=2_nosplit'

if ${train} || ${run_all} ; then
    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='imitate' \
    +experiment=mountain_car/bc/train \
    imitation/imitator=bce-disc-2p \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=True \
    ++imitation.imitator.instance.split_demos=false \
    ++imitation.vars.demo_file_pattern=${demo_file_pattern} \
    ++imitation.vars.demo_paths=${demo_paths}
fi


if ${eval} || ${run_all} ; then
    run_id=rid0 # Update to match rid generated by train

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

# CPU Replication
# Replicate original results by setting specific seeds and using cpu
######################################################################################
if true || ${run_all} ; then
train=true # Enable training
eval=true # Enable eval
algo_name=bc
data_name=backward-70%

if ${train} || ${run_all} ; then
    HYDRA_FULL_ERROR=1 WANDB_MODE=$wandb python main.py ${hydra_job} -cn='imitate' \
    +experiment=mountain_car/bc/train \
    imitation/imitator=bc-disc \
    ++output_dir.exp_group=${exp_group} \
    ++output_dir.algo_name=${algo_name} \
    ++output_dir.data_name=${data_name} \
    ++output_dir.run_id='cpu-replication' \
    ++vars.set_seed.seed=0 \
    ++vars.set_seed.cuda_deterministic=True \
    ~imitation.imitator.train_kwargs.n_batches \
    ++imitation.imitator.train_kwargs.n_epochs=500 \
    ++imitation.imitator.train_kwargs.log_interval=null \
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