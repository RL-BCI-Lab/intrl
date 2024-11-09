import functools
import os
import re
import pickle
import random
from pathlib import Path
from copy import deepcopy
from typing import List, Optional, Callable, Union
from pdb import set_trace
from os.path import join

import torch as th
import numpy as np
from stable_baselines3.common import utils, policies

from intrl.common.logger import logger

try:
    import tensorflow as tf
except ModuleNotFoundError:
    pass


def set_memory_growth():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


def set_tf_random_seed(seed: int = None, cuda_deterministic: bool = False) -> int:
    """
    Seed the different random generators and generate seed if none is passed.

        Args:
            seed: Seed to be used when setting random generators. If it is none, a random
                seed will be generated.
                
            cuda_deterministic: If a GPU is being used, enabling this allows for CuDNN seeding
                but can hinder performance.
    """
    seed = int.from_bytes(os.urandom(4), byteorder="big") if seed is None else seed
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for tf
    tf.random.set_seed(seed)

    if cuda_deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        
    return seed

def set_torch_random_seed(seed: int = None, cuda_deterministic: bool = False) -> int:
    """
    Seed the different random generators and generate seed if none is passed.

        Args:
            seed: Seed to be used when setting random generators. If it is none, a random
                seed will be generated.
                
            cuda_deterministic: If a GPU is being used, enabling this allows for CuDNN seeding
                but can hinder performance.
    """
    seed = int.from_bytes(os.urandom(4), byteorder="big") if seed is None else seed
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(seed)

    if cuda_deterministic:
        # Deterministic operations for CuDNN, it may impact performances
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        
    return seed


def random_choice(a, seed: int = None, **kwargs):
    if seed is None:
        seed = np.random.get_state()[1][0]
    rng = np.random.default_rng(seed=seed)
    
    return rng.choice(a, **kwargs)


def melt(nested_list):
    return _melt(nested_list)


def _melt(x, merged=[]):
    for i in x:
        if isinstance(i, list):
            melt(i)
            continue
        merged.append(i)
    return merged


def index(indexable, indexes):
    """ Index nested objects
    
        This method is akin to indexing a List n number of times. For example, a list
        A = [[1], [2]] could be indexed as A[0][0] to get 1. Likewise, one could call
        index(A, [0, 0]) to ge the same output.
        
        Args:
            Indexable: A indexable object (e.g., a list) that which contains other 
                indexable objects equal to or less than the length of indexes.
                
            indexes: A list of indexes for hierarchically indexing the passed indexable.
    """
    indexed = indexable
    for i, idx in enumerate(indexes):
        
        if isinstance(indexed, list) and idx >= len(indexed):
            msg = f"Index {idx!r} at iteration {i} does not exist for {indexable}"
            raise ValueError(msg)
    
        try:
            indexed = indexed[idx]
        except Exception:
            try:
                indexed = getattr(indexed,idx)
            except AttributeError:
                msg = f"Can not index object {indexed} by {idx}"
                raise IndexError(msg)
        
    return indexed


def get_one_hot(targets, n_classes):
    res = np.eye(n_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[n_classes])


def load_files(
    paths: List[str], 
    load_func: Callable,
    choose_func: Optional[Callable] = lambda files: files,
    pattern: Optional[str] = None, 
    throw_error: Optional[bool] = True, 
    label_data: Optional[bool] = False,
    verbose: Optional[bool] = False,
):
    """ Load files using an absolute path or search and load files using specified patter 
    
        Args:
            paths: A list of strings of absolute paths or path to directories where the
                pattern arg will be used to search for files.

            load_func: A function or callable that takes in a path and returns the
                data loaded from the file. Data should be returned as a list.
                
            choose_func: A function that takes in the found_files and returns
                which files to use. By default all found files are used. Callable
                should take in a iterable of strings and return an iterable of strings.

            pattern: A regex pattern that will be searched if a path is not a absolute
                path to a file.

            throw_error: If true, all warnings will be thrown as errors.

            label_data: A list of paths corresponding to each element in data.
    """
    data = []
    labels = []
    found_files = choose_func(
        locate_files(paths=paths, pattern=pattern, throw_error=throw_error)
    )
    for f in found_files:
        d = load_func(f)
        data.extend(d)
        if label_data: labels.extend([f]*len(d))

    if throw_error and len(data) == 0:
        msg = f"No data was found or loaded from {paths}."
        raise ValueError(msg)
    if verbose:
        msg = f"Loaded {len(data)} files from paths {[paths]}"
        logger.info(msg) if logger.CURRENT else print(msg)
        
    if label_data:
        return data, labels
    return data


def locate_files(
    paths: List[str], 
    pattern: Optional[str] = None,
    throw_error: Optional[bool] = True, 
):
    if isinstance(paths, str):
        paths = [paths]
    found = []
    
    for path in paths:
        if os.path.isfile(path):
            found.append(path)
        else:
            if pattern is None:
                msg = "If a given path is not a file, 'pattern' must be specified."
                raise ValueError(msg)
            found.extend(search_directories(path, pattern, throw_error))
            
    if len(found) == 0:
        msg = f"No files were found at the paths {paths}."
        raise ValueError(msg)    
    
    return found
 
def alphanum_key(s: str):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
        
        Ref: https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
    """
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    return [tryint(c) for c in re.split('([0-9]+)', str(s))]


def search_directories(
    paths: List[str],
    pattern: str, 
    throw_error: Optional[bool] = True,
    sort: bool = True,
    sort_func: Callable = alphanum_key
):
    if isinstance(paths, str):
        paths = [paths]
        
    found = []
    pattern = re.compile(pattern)
    for path in paths:
        f = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")
        for root,dirs,files in os.walk(path):
            for p in dirs+files:
                # Search entire path incase pattern contains parent directories
                cur_path = join(root, p)
                if pattern.search(cur_path):
                    f.append(cur_path)
                    
        if throw_error and len(f) == 0:
            msg = f"No files were found using pattern {pattern!r} in {path!r}."
            raise ValueError(msg)
        found.extend(f)
   
    return sorted(found, key=sort_func)   


def find_and_load_state_dicts(
    paths: List[str], 
    model: Union[policies.BaseModel, th.nn.Module],
    pattern: Optional[str] = '.*\.pt', 
    device: Union[th.device, str] = "auto",
    throw_error: Optional[bool] = True, 
    label_data: Optional[bool] = False,
    verbose: Optional[bool] = False
):
    """Load torch module state dictionary from specified path using a specific file pattern.
    
        If multiple files are found, all files will attempt to be loaded to copies of 
        the same model.

    Args:
        path: path to directory in which policy will be saved
    """
    def load(path):
        if hasattr(model, 'load'):
            m = model.load(path, utils.get_device(device))
        else:
            m = deepcopy(model)
            state_dict = th.load(path, map_location=utils.get_device(device))
            m.load_state_dict(state_dict)
        if verbose:
            msg = f"Loading model from {path}"
            logger.info(msg) if logger.CURRENT else print(msg)

        return [m]

    return load_files(
        paths=paths,
        pattern=pattern,
        load_func=load,
        throw_error=throw_error,
        label_data=label_data
    )


def rgetattr(obj, path: str, delim='.', *default):
    """
    :param obj: Object
    :param path: 'attr1.attr2.etc'
    :param default: Optional default value, at any point in the path
    :return: obj.attr1.attr2.etc
    """
    attrs = path.split('.')
    try:
        return functools.reduce(getattr, attrs, obj)
    except AttributeError:
        if default:
            return default[0]
        raise


def reduce_dims(data, ordered_pairs, allow_1D=False, reshape_order='C'):
    """ Reduces (reshapes) dimensions of data based on pairing of dimensions
        Args:
            data: Torch tensor or numpy array which will be reduced using reshaping.
            
            ordered_pairs: A list of sets where each sub-list represents
                the new grouping of dims. Order is assumed such that the new
                dimensions will reflect the order of the ordered_pairs.
                
            allow_1D: If False and shape of data is 1D then a new dimension is added
                to the end of the data to keep shape 2D. If True, this is disabled.
    """
    new_dims = []
    shape = np.array(data.shape)
    shape_idx = {i for i in range(len(shape))}
    used_shape_idx = set()

    for pair in ordered_pairs:
        # Track which indexes have been used and check they are valid
        pair_set = set([pair]) if isinstance(pair, int) else set(pair) 
        inter = pair_set.intersection(shape_idx)
        if inter.intersection(used_shape_idx):
            raise ValueError(f"order_pairs {ordered_pairs} contains duplicate dims.")
        if len(inter) == 0:
            raise ValueError(f"Dimensions {pair} do not exists.")
        used_shape_idx |= inter
        
        # Compute new dimension size
        new_dim_size = shape[pair] if isinstance(pair, int) else shape[pair].prod()
        new_dims.append(new_dim_size)

    # Check if all indices were given, if not add them to the end
    diff = shape_idx.difference(used_shape_idx)
    if diff != 0:
        new_dims.extend(shape[list(diff)])
    
    # Keep shape 2D
    if not allow_1D and len(new_dims) == 1:
        new_dims.append(1)

    return data.reshape(*new_dims, order=reshape_order)