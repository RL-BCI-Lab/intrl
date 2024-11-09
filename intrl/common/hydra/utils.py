import os
import ast
import re
import warnings
from typing import Any, List
from os.path import join
from pathlib import Path
from pdb import set_trace

import yaml
import numpy as np
from omegaconf import OmegaConf
from omegaconf.resolvers.oc.dict import _get_and_validate_dict_input
from hydra.utils import get_class, get_method, get_original_cwd

from intrl.common.utils import rgetattr


class DetermineRunID():
    """ OmegaResolver for Hydra to automatically determine the current run ID for experiments. """
    def __init__(self, prefix=None):
        self.prefix = prefix
    
    def __call__(self, path, check_cache=True, ids_to_exist=[]):
        run_id = self._get_proper_run_id(path, ids_to_exist)
        return int(run_id)

    def _get_proper_run_id(self, save_path, ids_to_exist):
        save_path = Path(save_path)
        prior_run_ids = []
        
        # Only check for prior ids if path exists
        if save_path.exists():
            prior_run_ids = self._find_prior_run_ids(save_path.iterdir())
        else:
            msg = f"Can not find prior ids as save path {str(save_path)!r} does not exist."
            warnings.warn(msg)

        # Combine prior ids and future ids to get all ids that will exist
        all_ids = np.unique(np.hstack([prior_run_ids, ids_to_exist]))
        # If no ids will exist the first id should be 0 then
        if len(all_ids) == 0:
            return 0
        # Find all the potential next IDs
        potential_next_ids = self._get_next_run_id(all_ids)
        # If no potential ids are found then +1 to the last ID
        # otherwise take the first potential ID (smallest next ID)
        if len(potential_next_ids) == 0:
            return all_ids[-1] + 1
        else:
            return potential_next_ids[0]
        
    def _find_prior_run_ids(self, folders):
        """Finds prior ids assuming last character is a digit"""
        prior_run_ids = []
        if self.prefix:
            for f in folders:
                if f.name.startswith(self.prefix):
                    # first element should be the prefix
                    id_text = f.name.split(self.prefix)[-1]
                    # If remaining text is not just a digit, ignore it
                    if id_text.isdigit():
                        prior_run_ids.append(int(id_text))
        else:
            prior_run_ids = np.array([int(f.name) for f in folders if str(f.name).isdigit()])
        return np.sort(np.array(prior_run_ids))

    def _get_next_run_id(self, all_ids):
        potential_ids = np.arange(0, all_ids[-1]+1)
        return np.setdiff1d(potential_ids, all_ids)

 
def get_run_id(path, prefix=None, ids_to_exist=[]):
    drid = DetermineRunID(prefix=prefix)
    run_id = drid(path, ids_to_exist)
    return run_id

from omegaconf import AnyNode, Container, DictConfig, ListConfig
from omegaconf.basecontainer import BaseContainer
from collections import OrderedDict
def order_dict_numeric_values(key: str, _root_: BaseContainer, _parent_: Container) -> ListConfig:
    assert isinstance(_parent_, BaseContainer)
    in_dict = _get_and_validate_dict_input(
        key, parent=_parent_, resolver_name="order_dict_numeric_values"
    )

    content = in_dict._content
    assert isinstance(content, dict)
    assert [int(k) for k in content.keys()]
    content = DictConfig(OrderedDict(
        sorted(
            content.items(), 
            key = lambda i: 0 if int(i[0]) == 0 else -1 / int(i[0]))
    ))
    
    ret = ListConfig([])
    if key.startswith("."):
        key = f".{key}"  # extra dot to compensate for extra level of nesting within ret ListConfig
    for k in content:
        ref_node = AnyNode(f"${{{key}.{str(k)!s}}}")
        ret.append(ref_node)

    # Finalize result by setting proper type and parent.
    element_type: Any = in_dict._metadata.element_type
    ret._metadata.element_type = element_type
    ret._metadata.ref_type = List[element_type]
    ret._set_parent(_parent_)

    return ret
   
def set_omega_resolvers():
    OmegaConf.register_new_resolver("join", lambda *args: join(*args), replace=True)
    OmegaConf.register_new_resolver("getattr", getattr, replace=True)
    OmegaConf.register_new_resolver("get_class", get_class, replace=True)
    OmegaConf.register_new_resolver("get_method", get_method, replace=True)
    OmegaConf.register_new_resolver("rgetattr", rgetattr, replace=True)
    OmegaConf.register_new_resolver("get_run_id", get_run_id, replace=True, use_cache=True)
    OmegaConf.register_new_resolver("getcwd", os.getcwd, replace=True)
    OmegaConf.register_new_resolver("get_original_cwd", get_original_cwd, replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("merge", OmegaConf.merge, replace=True)
    OmegaConf.register_new_resolver("order_dict_numeric_values", order_dict_numeric_values, replace=True)


def get_yaml_loader():
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u"""^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""", re.X),
        list(u'-+0123456789.'))
  
    def eval_constructor(loader, node):
        """ Extract the matched value, expand env variable, and replace the match """
        value = node.value

        if isinstance(value, str):
            match = "".join(eval_matcher.findall(value))
            return eval(match)
        
    def tuple_constructor(loader, node):                                                               
        def parse_tuple(element):                                          
            if element.isdigit(): 
                return int(element)
            try:
                return float(element)
            except ValueError:
                pass 
            try:
                if ast.literal_eval(value) is None:
                    return None
            except ValueError:
                return value

        value = loader.construct_scalar(node)
        # Match tuple(*) and remove it from string. At the same time strip any whitespace
        # and split string into list based on commas.                                                                                                                                                    
        match = "".join(tuple_matcher.findall(value)).replace(' ', '').split(',')
        # Remove tailing space if tuple was formated with tailing comma tuple(*,)                                                                                                                                                   
        if match[-1] == '':                                                                                                       
            match.pop(-1)
        # Convert string to int, float, or string.                                                                                      
        return tuple(map(parse_tuple, match))                                                       
    
    def none_constructor(loader, node):
        """ Extract the matched value, expand env variable, and replace the match """
        value = node.value

        if isinstance(value, str):
            try:
                if ast.literal_eval(value) is None:
                    return None
            except ValueError:
                return value
        

    eval_matcher = re.compile(r'eval\(([^}^{]+)\)')
    loader.add_implicit_resolver('!eval', eval_matcher, None)
    loader.add_constructor(u'!eval', eval_constructor)

    tuple_matcher = re.compile(r'\(([^}^{]+)\)')
    loader.add_implicit_resolver('!tuple', tuple_matcher, None)
    loader.add_constructor(u'!tuple', tuple_constructor)

    none_matcher = re.compile(r'None')
    loader.add_implicit_resolver('!none', none_matcher, None)
    loader.add_constructor(u'!none', none_constructor)
    
    return loader

def get_yaml_dumper():
    dumper = yaml.Dumper
    
    def ndarray_rep(dumper, data):
        return dumper.represent_list(data.tolist())

    dumper.add_representer(np.ndarray, ndarray_rep)
    
    return dumper

def load_yaml(yaml_path, loader=None, **kwargs):
    """ Loads a yaml file
    
        Args:
            yaml_path (str): File path to yaml file.
            
            loader (yaml.Loader): yaml loader to be used for loading. Will default
                to custom SafeLoader.
            
            kwargs (dict): Arguments for get_yaml_loader() function which
                creates the custom SafeLoader. 

        Returns:
            dict: Returns a parsed dictionary extracted from the yaml file.
    """
    
    if not os.path.exists(yaml_path):
        raise ValueError("Path to config does not exist {}".format(yaml_path))
    
    loader = get_yaml_loader(**kwargs) if loader is None else loader
    with open(yaml_path, 'r') as stream:
        params = yaml.load(stream, Loader=loader)

    return params

def dump_yaml(data, yaml_path, **kwargs):
    with open(yaml_path, 'w') as stream:
        yaml.dump(data, stream, Dumper=get_yaml_dumper(), **kwargs)