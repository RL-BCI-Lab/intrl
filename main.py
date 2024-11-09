import sys
import os
import warnings
import pickle
from os.path import join
from datetime import datetime
from abc import ABC, abstractmethod
from typing import (
    Optional,
    Union,
    Dict
)
from pdb import set_trace

# pipeline_logger() imports
from functools import wraps
from time import time
from datetime import timedelta

import wandb
import torch
import hydra
from hydra.utils import get_original_cwd, instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from stable_baselines3.common import utils
from stable_baselines3.common.logger import Logger as SBLogger

from intrl.common.hydra.utils import set_omega_resolvers
from intrl.common.data.storage import H5Storage
from intrl.common.logger import logger as global_logger
from intrl.common.logger import ShadowLogger
from intrl.common.data import rollout
from intrl.common.data.replay import Replayer


def pipeline_logger(logger: Union[SBLogger, ShadowLogger], name: str):
    """ Decorator for logging execution time for pipelines 
    
        Args:
            logger: A ShadowLogger or SBLogger for logging
            
            name: Name of the pipeline stage that will be used for logging.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Starting: {name}")
            start_time = time()
            results = func(*args, **kwargs)
            end_time = time()-start_time
            logger.info(f"Finished: {name}")
            logger.info(f"Execution time for {name!r}: {str(timedelta(seconds=end_time))}")
            return results
        return wrapper
    return decorator


class SequentialPipeline(ABC):
    """ Pipeline class for building the IntRL Hydra-core configs in stages.
        
        Class works by taking a hydra formatted config which will be build in stages. 
        After a sub-dictionary (i.e., stage) is "built" any sub-dictionary thereafter
        will have access to these variables. If an object is built, following stages
        will have to reference the object attributes NOT the dictionary path. This can
        be done using the resolver ${getattr:} in the hydra-config.

        Config default stage build order:
            1) output_dir
            2) vars 
            
        Within stages, build order is given as below. Note, output_dir, vars and env_spec 
        do not contain the below keys:
            1) vars
            2) logger
            3) any remaining keys (usually specified under a single unique key)\
        
        Attributes:
            cfg: The raw config which contains all the parameters for running the pipeline.
                This should be an OmegaConf config (i.e., hydra config)
                
            built_cfg: The instantiated version of the self.cfg which is built a specific
                stages.
    """
    def __init__(self, config):
        self.cfg = config
        self.built_cfg = None

    def run(self):
        # Build output directory
        output_dir = OmegaConf.to_container(self.cfg.output_dir,resolve=True)
        self._build_init_cfg(OmegaConf.create({'output_dir': output_dir}))
        # Build global variables
        self._instantiate('vars', exclude_vars=True)
        self._run_pipeline_stages()

    @abstractmethod
    def _run_pipeline_stages(self):
        """Additional pipeline stages to run after default stages"""
        pass
    
    def _instantiate(self, key: str, exclude_vars: bool = False):
        """ Instantiates the specified sub-dictionary. 
        
            When being instantiated, 'vars' are built first, followed by 'logger',
            and then the remaining top level keys are built together. Skip can be
            passed to the 'vars' to skip building this sub-dictionary entirely.
            
            Args:
                key: The key or name of the sub-dictionary being built.
                
                exclude_vars: Excludes the building of 'vars'.
                
            Returns:
                An instantiated dictionary for the passed key.
        """
        sub_cfg = OmegaConf.create({key: self.cfg[key] or {}})
        # Build current stage's vars (if given)
        if 'vars' in sub_cfg[key]:
            if sub_cfg[key].vars:
                # If skip is true, dont build or add any variables to built_cfg
                # for given stage.
                if 'skip' in sub_cfg[key].vars and sub_cfg[key].vars.skip:
                    return None
                
                self._build_init_cfg( 
                    OmegaConf.create({key: {'vars': sub_cfg[key].pop('vars')}})
                )
        elif not exclude_vars:
            raise KeyError(f"Stage {key} in config is missing key 'vars'")
            
        # Build current stage's logger (if given)
        if 'logger' in sub_cfg[key] and sub_cfg[key].logger:
            self._build_init_cfg(
                 OmegaConf.create({key: {'logger': sub_cfg[key].pop('logger')}})
            )
        
        self._build_init_cfg(sub_cfg)
        if self.built_cfg[key] is None:
            msg = f"The stage key {key!r} has been set to None. This may cause " \
                   "errors if other stages refer to this stage's config later on."
            global_logger.warn(msg) 
            
        return self.built_cfg[key]
    
    def _build_init_cfg(self, cfg: DictConfig, **kwargs):
        """ Builds a config using Hydra's instantiate() function
        
            Args:
                cfg: Config to be instantiated.
                
                kwargs: Additional keyword arguments for the instantiate() function.
        """
        if self.built_cfg is None:
            self.built_cfg = instantiate(cfg,  _convert_='none', **kwargs)
        else:
            self.built_cfg = instantiate(cfg,  _convert_='none', **self.built_cfg, **kwargs)
            
    def init_wandb(self, save_dir: str, stage: str):
        """ Initializes weights and biases for experiment tracking 
        
            Args:
                save_dir: Directory to save wandb info to
                
                stage: Name of stage to be used when creating job name
        """
        group_name = join(
            self.cfg.output_dir.exp_group,
            self.cfg.output_dir.env_name,
        )
        job_type = self.cfg.output_dir.data_name
        
        job_name = join(
            self.cfg.output_dir.algo_name,
            # Hydra changes cwd to the current rid, self.cfg.output_dir.run_id is
            # built dynamically using current cwd, thus it will return the wrong 
            # ID if called here!
            os.path.basename(os.getcwd()),
            stage
        )
        wandb.init(
            project='intrl', 
            dir=save_dir, 
            group=group_name,
            job_type=job_type, 
            name=job_name,
            config=dict(self.cfg),
        )

class ReplayPipeline(SequentialPipeline):
    """ Pipeline for replaying recorded demonstrations 
        
        replayer.instance is typically of type intrl.common.data.replay.Replayer but
        duck-typing can be used if a play() method is implemented.
        
        Config default stage build order:
            1) output_dir
            2) vars 
            3) env_specs
            4) replay
    """
    def __init__(self, config):
        super().__init__(config)
    
    def _run_pipeline_stages(self):
        self._run_replay()
        
    @pipeline_logger(logger=global_logger, name='Stage - Replay')
    def _run_replay(self):
        replay = self._instantiate('replay')
        
        vars_, replayer, logger = replay.vars, replay.replayer, replay.logger
        
        if not  isinstance(replayer.instance, Replayer):
            msg = f"replayer.instance {type(replayer)!r} must be of type Replayer"
            raise ValueError(msg)
        
        @pipeline_logger(logger=global_logger, name='Play')
        def play():
            if hasattr(replayer.instance, 'fps'): 
                logger.info(f"Replaying at {replayer.instance.fps} FPS")
            play_kwargs = replayer.play_kwargs or {}
            replayer.instance.play(**play_kwargs)
        play()
        
        
class CollectPipeline(SequentialPipeline):
    """ Pipeline for collecting demonstrations
    
        collect_traj.collector can be a function or class with the __call__ method. A
        List[Trajectory] must be returned.
        
        Config default stage build order:
            1) output_dir
            2) vars 
            3) env_specs
            4) collect
            5) save
    """
    def __init__(self, config):
        super().__init__(config)
  
    def _run_pipeline_stages(self):
        self._instantiate('env_spec')
        # self._run_train_expert()
        self._run_collect_trajectories()
        
    # TODO: Decouple from this pipeline into its own pipeline
    # @pipeline_logger(logger=global_logger, name='Stage - Expert Training')
    # def _run_train_expert(self):
    #     expert = self._instantiate('train_expert')
    #     # If none was returned, stage has been skipped.
    #     if expert is None:
    #         global_logger.info("Skipping stage...")
    #         return
    #     vars_, logger, agent = expert.vars, expert.logger, expert.agent
        
    #     @pipeline_logger(logger=logger, name='Training Expert')
    #     def train_expert():
    #         logger.set_logger(logger)
    #         agent.instance.learn(**agent.learn_kwargs)
    #     train_expert()
        
    @pipeline_logger(logger=global_logger, name='Stage - Collect Trajectories')
    def _run_collect_trajectories(self):
        ct = self._instantiate('collect_trajectories')
        if ct is None:
            global_logger.info("Skipping stage...")
            return
        
        @pipeline_logger(logger=ct.logger, name='Trajectory Collection')
        def collect():               
            trajectories = ct.collector.func()
            stats = rollout.log_rollout_stats(trajectories, logger=ct.logger)
            ct.logger.info(f"Rollout stats: {stats}")
            return trajectories
    
        @pipeline_logger(logger=ct.logger, name='Saving Environment Specs')
        def save_env_specs():
            with open('env.pickle', 'wb') as f:
                env_specs = dict(
                    observation_space=self.built_cfg.env_spec.env.instance.observation_space,
                    action_space=self.built_cfg.env_spec.env.instance.action_space,
                )
                pickle.dump(env_specs, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        @pipeline_logger(logger=ct.logger, name='Saving Trajectories')
        def save_trajectories(trajectories):
            ct.save.func(
                trajectories=trajectories, 
                **ct.save.kwargs
            )
        trajectories = collect()
        if ct.save is not None:
            save_env_specs()
            save_trajectories(trajectories)
        else: 
            ct.logger.info(f"Save skipped...")

class ImitationPipeline(SequentialPipeline):
    """ Pipeline for running imitation learning algorithms.
    
        imitator.instance is a class of type imitation.algorithms.base.DemonstrationAlgorithm
        but duck-typing can be used if a train() and save_policy() method are implemented.
        
        Config default stage build order:
            1) output_dir
            2) vars 
            3) env_specs
            4) imitation

    """
    def __init__(self, config):
        super().__init__(config)
    
    def _run_pipeline_stages(self):
        self._instantiate('env_spec')
        self._run_imitation_learning()
    
    @pipeline_logger(logger=global_logger, name='Stage - Imitation')
    def _run_imitation_learning(self):
        imit = self._instantiate('imitation')
        if imit is None:
            msg = "Imitator not initialized"
            global_logger.error(msg)
            raise ValueError(msg)
        vars_, logger, imitator = imit.vars, imit.logger, imit.imitator
       
        @pipeline_logger(logger=logger, name='Training Imitator')
        def train_imitator():
            if vars_.use_wandb: self.init_wandb(logger.dir, 'train')
            train_kwargs = imitator.train_kwargs or {}
            imitator.instance.train(**train_kwargs)
            if vars_.use_wandb: wandb.finish()
            
        @pipeline_logger(logger=logger, name='Saving Imitator Policy')
        def save_policy():
            os.makedirs(os.path.dirname(vars_.save_path), exist_ok=True)
            if hasattr(imitator.instance, 'save_policy'):
                imitator.instance.save_policy(vars_.save_path)
            elif hasattr(imitator.instance, 'policy'):
                torch.save(imitator.instance.policy, vars_.save_path)
            else: 
                warnings.warn("Unable to save policy using 'policy' attribute or 'save_policy()' method")

        train_imitator()
        if vars_.save_path is not None:
            save_policy()
        else:
            logger.info(f"Training skipped...")


class EvalPipeline(SequentialPipeline):
    """ Pipeline for evaluating RL algorithms 
    
        evaluator.func can be a function or class with a __call__() method. If the function
        requires arguments, pass a partial version of the function with the arguments set.
        
        Config default stage build order:
            1) output_dir
            2) vars 
            3) env_specs
            4) evaluation

    """
    def __init__(self, config):
        super().__init__(config)
        
    def _run_pipeline_stages(self):
        self._instantiate('env_spec')
        self._run_evaluation()
        
    @pipeline_logger(logger=global_logger, name='Stage - Evaluation')
    def _run_evaluation(self):
        evl = self._instantiate('evaluation')
        if evl is None:
            msg = "Evaluator not initialized"
            global_logger.error(msg)
            raise ValueError(msg)
        
        @pipeline_logger(logger=evl.logger, name='Running Evaluator')
        def run_evaluator():
            # self.init_wandb(logger.dir, 'eval')
            trajectories = evl.evaluator.func()
            stats = rollout.log_rollout_stats(trajectories, logger=evl.logger)
            evl.logger.info(f"Rollout stats: {stats}")
            # wandb.finish()
        run_evaluator()


@hydra.main(version_base=None, config_path='configs/')
def main(config):
    """ Main function loading the hydra-config and selecting pipeline to run. """
    hydra_cfg = HydraConfig.get()
    config_name  = hydra_cfg.job.config_name
    assert hydra_cfg.run.dir == config.output_dir.hydra_run_dir, \
            f"Path mismatch between {hydra_cfg.run.dir} and {config.output_dir.hydra_run_dir}"

    now = datetime.now()
    global_logger.info(
        f"Executing on {now.strftime('%d/%m/%Y')} at {now.strftime('%H:%M:%S')}"
    )
    global_logger.info(f"Current working directory: {os.getcwd()}")
    global_logger.info(f"Original working directory: {get_original_cwd()}")
    
    if config_name == 'collect':
        pipeline = CollectPipeline(config)
    elif config_name == 'replay' or config_name == 'feedback':
        pipeline = ReplayPipeline(config)
    elif config_name == 'imitate':
        pipeline = ImitationPipeline(config)
    elif config_name == 'eval':
        pipeline = EvalPipeline(config)
    else:
        msg = f"Config name {config_name!r} has no corresponding pipeline."
        raise ValueError(msg)

    @pipeline_logger(logger=global_logger, name='Pipeline')
    def run_pipeline():
        pipeline.run()
    run_pipeline()
    
    now = datetime.now()
    global_logger.info(
        f"Finished executing on {now.strftime('%d/%m/%Y')} at {now.strftime('%H:%M:%S')}"
    )

if __name__ == '__main__':
    set_omega_resolvers()
    main()