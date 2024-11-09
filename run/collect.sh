#!/bin/bash

######################################################################################
# Naming Scheme
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

# TEST
######################################################################################
if false ; then
experiment=mountain_car/human-collect
exp_group=tests
algo_name=human
data_name=test
min_episodes=3

HYDRA_FULL_ERROR=1 python main.py ${hydra_job} -cn='collect' \
+experiment=${experiment} \
++output_dir.exp_group=${exp_group} \
++output_dir.algo_name=${algo_name} \
++output_dir.data_name=${data_name} \
++collect_trajectories.collector.func.sample_until.min_episodes=${min_episodes}
fi
# BACKWARDS
######################################################################################
if false ; then
experiment=mountain_car/human-collect
exp_group=v0
algo_name=human
data_name=backwards
min_episodes=10

HYDRA_FULL_ERROR=1 python main.py ${hydra_job} -cn='collect' \
+experiment=${experiment} \
++output_dir.exp_group=${exp_group} \
++output_dir.algo_name=${algo_name} \
++output_dir.data_name=${data_name} \
++env_spec.vars.env_wrappers.2.kwargs.decreasing=true \
++collect_trajectories.collector.func.sample_until.min_episodes=${min_episodes}
fi
# BACKWARDS NOISE
# Backwards policy but does not reach the goal
######################################################################################
if false ; then
experiment=mountain_car/human-collect
exp_group=v0
algo_name=human
data_name=backwards-noise
min_episodes=10

HYDRA_FULL_ERROR=1 python main.py ${hydra_job} -cn='collect' \
+experiment=${experiment} \
++output_dir.exp_group=${exp_group} \
++output_dir.algo_name=${algo_name} \
++output_dir.data_name=${data_name} \
++env_spec.vars.env_wrappers.2.kwargs.decreasing=true \
++collect_trajectories.collector.func.sample_until.min_episodes=${min_episodes}
fi
# FORWARDS
######################################################################################
if false ; then
experiment=mountain_car/human-collect
exp_group=v0
algo_name=human
data_name=forwards
min_episodes=10

HYDRA_FULL_ERROR=1 python main.py ${hydra_job} -cn='collect' \
+experiment=${experiment} \
++output_dir.exp_group=${exp_group} \
++output_dir.algo_name=${algo_name} \
++output_dir.data_name=${data_name} \
++env_spec.vars.env_wrappers.2.kwargs.decreasing=false \
++collect_trajectories.collector.func.sample_until.min_episodes=${min_episodes}
fi

######################################################################################
# Safety Gym
######################################################################################

# TEST
######################################################################################
if true ; then
experiment=mujoco/safety/human-collect
exp_group=tests
algo_name=human
data_name=avoid-area
min_episodes=3

HYDRA_FULL_ERROR=1 python main.py ${hydra_job} -cn='collect' \
+experiment=${experiment} \
++output_dir.exp_group=${exp_group} \
++output_dir.algo_name=${algo_name} \
++output_dir.data_name=${data_name} \
++collect_trajectories.collector.func.sample_until.min_episodes=${min_episodes}
fi
# Avoid Area Bottom
######################################################################################
if false ; then
experiment=mujoco/safety/human-collect
exp_group=v0
algo_name=human
data_name=avoid-area/bottom
min_episodes=10

HYDRA_FULL_ERROR=1 python main.py ${hydra_job} -cn='collect' \
+experiment=${experiment} \
++output_dir.exp_group=${exp_group} \
++output_dir.algo_name=${algo_name} \
++output_dir.data_name=${data_name} \
++collect_trajectories.collector.func.sample_until.min_episodes=${min_episodes}
fi

# OLD EXAMPLES (OUTDATED)
######################################################################################
# HYDRA_FULL_ERROR=1 python main.py -m -cn=collect \
# +experiment=mountain_car/human-collect \
# ++output_dir.algo_name=human/test \
# ++collect_trajectories.collector.func.sample_until.min_episodes=1 \
# ++collect_trajectories.vars.extract_info_keys=['render']

# HYDRA_FULL_ERROR=1 python main.py -m -cn=collect \
# +experiment=mountain_car/human-collect \
# ++output_dir.job_name=test \
# ++collect_trajectories.collector.func.sample_until.min_episodes=1 \
# ++collect_trajectories.vars.save_kwargs.extract_info_keys=['render'] \
# ++collect_trajectories.collector.func.policy.fps=60

# HYDRA_FULL_ERROR=1 python main.py -m -cn=collect \
# +experiment=mountain_car/human-collect \
# ++output_dir.job_name=\
# "$\{.env_name\}/$\{.algo_name\}/nothing/bad",\
# "$\{.env_name\}/$\{.algo_name\}/nothing/bad",\
# "$\{.env_name\}/$\{.algo_name\}/nothing/bad",\
# "$\{.env_name\}/$\{.algo_name\}/nothing/bad",\
# "$\{.env_name\}/$\{.algo_name\}/nothing/bad"
