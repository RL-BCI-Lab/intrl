
# IntRL

A repository tackling all aspects interactive RL algorithms from data collection to algorithm training and evaluation. This repository is built on top of [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3), [Imitation](https://github.com/HumanCompatibleAI/imitation), and [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

# Installation

## Conda Environment
The provided conda environment can be built using the following command:

```
conda env create -f docker/full-env.yaml 
```

If you do not wish to build the conda environment, view the `docker/full-env.yaml` for
dependencies.

Once created, the `intrl` module needs to be installed by running the following command.

```
pip install -e .
```

## Docker Environment (WIP)
The GPU Docker environment has two requirements: 1) you will need to use a version of Linux where [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is supported or WSL 2.0 with Windows 11 can also work; 2) you need to have a GPU and the latest Nvidia drivers installed on your host PC. If you meet these requirements then you can use the Docker environment with a GPU provide below.

*For the following commands, we will assume you are using Linux or WSL 2.0*

### Running Images 
There are pre-defined commands for creating the IntRL docker container. To access these commands simply source the `env.sh` script within  `docker/` directory.  

```
source docker/env.sh
```
Once done, you will have access to the alias commands `intrl-gpu-make` which will create a docker container that has access to any local GPUs.

### Attaching
After running the `intrl-gpu-make` command you will need to attach to your container. You can do so by doing a `docker attach` followed by the name of the container `intrl-<version>`. The `<version>`is determined by the tag of the docker image. For now, the default is `torch`. An example command for attaching to the GPU container with `torch` version is given below. *Note, if you are not running multiple containers at once a tab complete will usually automatically find the name of the container.*

```
docker attach intrl-torch
```

# Quickstart
 To run the code, Hydra is used to orchestrate the building of configs. Once a config is built, it is passed to particular pipeline where its contents are used by the pipeline to run the code. Each pipeline has one or more corresponding Hydra templates located in the root of the `config/` directory. The current pipelines are listed below.

 - collect
    - Collects demonstrations from either a human or agent. These demonstrations are then saved, by default, to the `demos/` directory. 
 - replay
    - Replays collected demonstrations for viewing purposes.
 - feedback
    - Replays collected demonstrations but allows for feedback to be captured and mapped to demonstration states.
 - imitation
    - Trains Imitation algorithms or algorithms which inherent from the `imitation.algorithms.base.DemonstrationAlgorithm` class.
 - evaluation
    - Evaluates imitation and RL algorithms using the provided evaluation function. 

## CLI (Manual Commands)
Manual commands can be done by running the `main.py` file using Hydra [CLI interface](https://Hydra.cc/docs/1.1/tutorials/basic/your_first_app/simple_cli/#internaldocs-banner). See the following template as an example:

```
python main.py -cn=<Hydra-template> +experiment=<config/experiment/>
```

For complex commands that require many overrides, it is recommended to use a bash scrip instead (see the following section).

## Bash Scripts (Predefined Commands)
To run predefined commands refer to the `run/` directory which contains bash scripts doing so. These bash scripts define more complex Hydra commands that override multiple variables to such specific experiments. Feel free to refer to these scripts to create your own or edit the existing scripts to fit your needs.

# Data Collection
Current data collection only allows for the capturing of demonstrations, the replaying of said demonstrations, and the capturing of feedback for previously captured demonstrations. The data collection interface replies on Gymnasium wrappers for controlling the game and PyGame for the interface. As such the wrappers and interface can fail depending on the type of game being used but tends to work for all officially implemented Gymnasium tasks, excluding continuous action tasks.

Running data collection test:
```
/bin/bash run/collect.sh
```

Running data replay test:
```
/bin/bash run/replay.sh
```

Running data feedback test:
```
/bin/bash run/feedback.sh
```

# Algorithms
The following is a list of implemented algorithms. On top of these, one can run algorithms from SB3 or Imitation using Hydra although these configs will constructed from scratch.

Running `intrl.algorithms.bc.bcensemble` test:
```
/bin/bash run/bcensemble/test.sh
```

Running `intrl.algorithms.bc.bcnoise` test:
```
/bin/bash run/bcnoise/test.sh
```

Running `intrl.algorithms.bc.bcshape` test:
```
/bin/bash run/bcshape/test.sh
```