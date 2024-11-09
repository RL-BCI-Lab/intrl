#!/bin/bash

# Collect human demos
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

# Replay human demos (easiest way is to replicate the existing path using env_name/algo_name/run_id)
########################################################################################
# HYDRA_FULL_ERROR=1 python main.py -m -cn=replay \
# +experiment=label-with-pygame \
# ++output_dir.env_name=MountainCar-v0 \
# ++output_dir.algo_name=human/test \
# ++output_dir.run_id=0 \

# HYDRA_FULL_ERROR=1 python main.py -m -cn=replay \
# +experiment=replay-with-pygame \
# ++output_dir.env_name=MountainCar-v0 \
# ++output_dir.algo_name=human/backward/bad \
# ++output_dir.run_id=0 \

# HYDRA_FULL_ERROR=1 python main.py -m -cn=replay \
# +experiment=noisy-replayer \
# ++output_dir.algo_name=human \
# ++output_dir.job_name=test \

# Train using various algorithms with test demo data
########################################################################################


### Classical BC using KLD loss
# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcshape/train-eval' \
# ++output_dir.algo_name='bcshape/test2/bc-using-kld' \
# ++output_dir.exp_group='vTest' \
# ++vars.set_seed.seed=0 \
# ++vars.set_seed.cuda_deterministic=True \
# ++imitation.imitator.instance.rng.seed=0 \
# ++imitation.imitator.instance.loss='kld' \
# ++imitation.imitator.instance.fb2tau.neg.value=1 \
# ++imitation.imitator.instance.fb2tau.pos.value=1 \
# ++imitation.imitator.instance.loss_kwargs.alpha_kld=0 \
# ++imitation.imitator.instance.loss_kwargs.alpha_nll=1 \
# ++imitation.imitator.instance.policy_specs.0.kwargs.logit_norm_class=Null \
# ++imitation.imitator.instance.policy_specs.0.kwargs.logit_norm_kwargs=Null \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/test/0"] \

###### BCEnsemble ######
# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcensemble/default' \
# ++output_dir.algo_name='bcensemble/test/bc' \
# ++output_dir.exp_group='vTest' \
# ++vars.set_seed.seed=0 \
# ++vars.set_seed.cuda_deterministic=True \
# ++imitation.imitator.instance.rng.seed=0 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/test/0"] \

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcensemble/train-eval' \
# imitation='bc/ensemble/standard-norm' \
# ++output_dir.algo_name='bcensemble/test/bc' \
# ++output_dir.run_id='standard-norm-1' \
# ++output_dir.exp_group='vTest' \
# ++vars.set_seed.seed=0 \
# ++vars.set_seed.cuda_deterministic=True \
# ++imitation.imitator.instance.rng.seed=0 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/test/0"] \

###### BCNOISE ######
## Reducing BCNoise to simple BC Test
# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bcnoise/test/bc' \
# ++output_dir.exp_group='vTest' \
# ++vars.set_seed.seed=0 \
# ++vars.set_seed.cuda_deterministic=True \
# ++imitation.imitator.instance.rng.seed=0 \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/test/0"] \

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bcnoise/test/bc' \
# ++output_dir.exp_group='vTest' \
# ++vars.set_seed.seed=0 \
# ++vars.set_seed.cuda_deterministic=True \
# ++imitation.imitator.instance.rng.seed=0 \
# ++imitation.imitator.instance.loss_kwargs.apply_temperature=False \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/test/0"] \

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bcnoise/test/bcnoise' \
# ++output_dir.exp_group='vTest' \
# ++vars.set_seed.seed=0 \
# ++vars.set_seed.cuda_deterministic=True \
# ++imitation.imitator.instance.rng.seed=0 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/test/0"] \

# Evaluate using BCNoise but such that the problem is reduced to just BC
########################################################################################
# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/eval' \
# ++output_dir.algo_name='bc' \
# ++output_dir.exp_group='vTest' \
# ++output_dir.run_id='train-30fps-eval-30fps-eval' \
# ++evaluation.evaluator.func.env.wrappers.1.kwargs.fps=30 \
# ++imitation.vars.load_path="$\{hydra:runtime.cwd\}/exps/vTest/MountainCar-v0/bc/train-30fps-eval-60fps/policy.pt"

# Evaulate: Below output_dir defines the path to the experiment for which to load by
# default. WARNING: This is will overwrite some files that were created when training!
########################################################################################
# HYDRA_FULL_ERROR=1 python main.py -cn=imitate \
# +experiment=mountain_car/bcnoise/eval \
# ++imitation.vars.load_path='$\{hydra:runtime.cwd\}/exps/vTest/$\{env_spec.vars.name\}/bc/forward/50g-50b/0/policy.pt' \
# ++imitation.imitator.instance.demonstrations=Null \
# ++imitation.imitator.vars.train=False \
# ++output_dir.algo_name=bc/forward/50g-50b \
# ++output_dir.exp_group=test \
# ++output_dir.run_id=0

# Train and evaluate BC (using BC Noise): Forward good, forward noisy
########################################################################################
# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bc/forward-noisy/70%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.vars.epsilon=0.7 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/"] \

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bc/forward-noisy/60%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.vars.epsilon=0.6 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/"] \

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bc/forward-noisy/50%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.vars.epsilon=0.5 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/"] \

# Train and evaluate BC (using BC Noise): Forward good, forward bad
########################################################################################
# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/forward/60%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/3"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/forward/50%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/4"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/forward/40%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/5"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/forward/30%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/6"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/forward/20%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/6",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/7"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/forward/10%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/6",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/7",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/8"]

# Train and evaluate BC (using BC Noise): Forward good, nothing (bad)
########################################################################################
# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/forward-nothing/60%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/forward-nothing/50%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/4"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/forward-nothing/40%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/5"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/forward-nothing/30%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/6"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/forward-nothing/20%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/6",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/7"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/forward-nothing/10%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/6",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/7",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/8"]

# Train and evaluate BC (using BC Noise): Backward good, backward noisy
########################################################################################
# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bc/backward-noisy/70%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.vars.epsilon=0.7 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/"] \

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bc/backward-noisy/60%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.vars.epsilon=0.6 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/"] \

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bc/backward-noisy/50%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.vars.epsilon=0.5 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/"] \

# Train and evaluate BC (using BC Noise): backward good, backward bad
#######################################################################################
# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/backward/60%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/3"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/backward/50%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/4"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/backward/40%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/5"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/backward/30%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/6"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/backward/20%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/6",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/7"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/backward/10%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/6",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/7",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/bad/8"]


# Train and evaluate BC (using BC Noise):backward good, nothing (bad)
########################################################################################
# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/backward-nothing/60%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/backward-nothing/50%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/4"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/backward-nothing/40%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/5"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/backward-nothing/30%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/6"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/backward-nothing/20%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/6",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/7"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bc/backward-nothing/10%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.train_kwargs.iterations=1 \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/6",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/7",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/8"]

# Train and evaluat BC Noise: Forward good, forward noisy
########################################################################################
# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bcnoise/forward-noisy/70%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.vars.epsilon=0.7 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/"] \

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bcnoise/forward-noisy/60%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.vars.epsilon=0.6 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/"] \

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bcnoise/forward-noisy/50%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.vars.epsilon=0.5 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/"] \

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bcnoise/forward-noisy/40%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.vars.epsilon=0.4 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/"] \

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bcnoise/forward-noisy/30%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.vars.epsilon=0.3 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/"] \

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bcnoise/forward-noisy/20%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.vars.epsilon=0.2 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/"] \

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bcnoise/forward-noisy/10%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.vars.epsilon=0.1 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/"] \


# Train and evaluate BC Noise: Forward good, Nothing
########################################################################################
# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bcnoise/forward-nothing/60%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bcnoise/forward-nothing/50%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/4"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bcnoise/forward-nothing/40%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/5"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bcnoise/forward-nothing/30%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/6"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bcnoise/forward-nothing/20%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/6",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/7"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bcnoise/forward-nothing/10%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/6",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/7",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/nothing/bad/8"]

# Train and evaluate BC Noise: Forward good, Foward bad
########################################################################################
# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bcnoise/forward/60%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/3"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bcnoise/forward/50%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/4"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bcnoise/forward/40%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/5"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bcnoise/forward/30%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/6"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bcnoise/forward/20%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/6",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/7"]

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-eval' \
# ++output_dir.algo_name='bcnoise/forward/10%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.imitator.instance.demonstrations.trajectories.paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/good/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/0",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/1",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/2",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/3",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/4",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/5",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/6",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/7",\
# "$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/forward/bad/8"]

# Train and evaluat BC Noise: Backward good, Backward noisy
########################################################################################
# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bcnoise/backward-noisy/70%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.vars.epsilon=0.7 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/"] \

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bcnoise/backward-noisy/60%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.vars.epsilon=0.6 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/"] \

# HYDRA_FULL_ERROR=1 python main.py -cn='imitate' \
# +experiment='mountain_car/bcnoise/train-noisy-eval' \
# ++output_dir.algo_name='bcnoise/backward-noisy/50%' \
# ++output_dir.exp_group='vTest' \
# ++imitation.vars.epsilon=0.5 \
# ++imitation.vars.traj_load_paths=\
# ["$\{vars.demo_dir\}/$\{env_spec.vars.name\}/human/backward/good/"] \

# Train and evaluate BC Noise: Backward good, nothing
########################################################################################


# Train and evaluate BC Noise: Backward good, backward bad
########################################################################################
