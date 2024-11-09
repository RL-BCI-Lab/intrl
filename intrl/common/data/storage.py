import warnings
import os
import io
import zipfile
from dataclasses import is_dataclass, fields
from pathlib import Path
from typing import (
    Optional,
    Mapping,
    Sequence,
    Iterable,
    List,
    Union,
    Dict,
    cast, 
    Any,
    Callable
)
from pdb import set_trace

import h5py
import pickle
import numpy as np
from imitation.data.types import AnyPath, Trajectory, Transitions

from intrl.common.utils import load_files, index
from intrl.common.logger import logger
from intrl.common.imitation.utils import extract_keys_from_trajectory_info

def load_trajectories(
    paths: Union[List[str], str], 
    traj_class: Trajectory = Trajectory,
    pattern: str = None, 
    label_data: bool = False,
    choose_func: Callable  = lambda files: files,
    throw_error: bool = True,
    use_infos: bool = False,
    keep_info_keys: List[str] = None,
    set_fields: Optional[Mapping[str, str]] = None,
    verbose: bool = False,
) -> Sequence[Trajectory]:
    """ Load all trajectories saved in a directory.
    
        Args:
            path: File path directory containing npz or h5 files
            
            pattern: Regex pattern of files to be loaded.
            
            label_data: Determines if load_files() returns a list of paths corresponding
                to the loaded files.
            
            choose_func: A function that takes in the found_files and returns
                which files to use. By default all found files are used. Callable
                should take in a iterable of strings and return an iterable of strings.
            
            throw_error: If true, all warnings will be thrown as errors when loading files.

            use_infos: Determines whether the info key will be loaded or not.
            
            set_fields: Merge a field stored in the npz file back into the info key. 
                Requires use_infos to be True.
                
            verbose: If true, verbose logging will be enabled when loading files.
        
        Returns:
            A list of trajectories either containing reward, feedback, or neither. This
            list is collected over all passed files.
        
    """
    def load_func(path):
        t = _load_trajectories(
            path,
            traj_class=traj_class,
            use_infos=use_infos,
            keep_info_keys=keep_info_keys,
            set_fields=set_fields,
        )
        if verbose:
            msg = f"Trajectories loaded from {path}: {len(t)}"
            logger.info(msg) if logger.CURRENT else print(msg)
        return t

    return load_files(
        paths=paths,
        load_func=load_func,
        choose_func=choose_func,
        pattern=pattern,
        label_data=label_data,
        throw_error=throw_error,
        verbose=verbose
    )


def _load_trajectories(
    path: AnyPath, 
    traj_class: Trajectory = Trajectory,
    use_infos: bool = False,
    keep_info_keys: List[str] = None,
    set_fields: Optional[List[str]] = None,
) -> Sequence[Trajectory]:
    """ Loads a sequence of trajectories given the provide file path.
    
        Args:
            path: File path to npz, h5, or huggingface dataset to be loaded.
            
            use_infos: Determines whether infos should be loaded or not.
            
            set_fields: Attempts to add additional keys from the infos group to
                the trajectory class. This can be useful when overwriting existing tuple
                information or adding missing tuple fields.
                
                Example: set_fields {
                            'feedback': 'fb-0',
                            'reward': 'reward'
                        }
            
                
        Returns:
            A list of trajectories either containing reward, feedback, or neither.
    """
    path = Path(path)
    if path.suffix in ['.h5']: 
        return H5Storage.load(
            path=path,
            traj_class=traj_class,
            use_infos=use_infos,
            keep_info_keys=keep_info_keys,
            set_fields=set_fields
        )
    elif path.suffix in ['.npz']:
        return NpzStorage.load(
            path=path,
            traj_class=traj_class,
            use_infos=use_infos,
            set_fields=set_fields
        )
    else:
        raise NotImplementedError


def load_pickle_file(path):
    with open(str(path), 'rb') as f:
         contents = pickle.load(f)
    return contents


def load_env_specs(
    paths: Union[List[str], str], 
    pattern: str = None, 
    label_data: bool = False,
    throw_error: bool = True,
    verbose: bool = False,
) :
    """ Loads the environment specifications ONLY from saved trajectory data.
    
        Args:
            path: File path directory containing npz files or huggingface dataset 
            
            pattern: Regex pattern of files to be loaded.
            
            label_data: Determines if load_files() returns a list of paths corresponding
                to the loaded files.
            
            throw_error: If true, all warnings will be thrown as errors when loading files.

            use_infos: Determines whether the info key will be loaded or not.
            
            verbose: If true, verbose logging will be enabled when loading files.
        
    """
    def load(path):
        t = load_pickle_file(path)
        if verbose:
            msg = f"Loaded env specs from {path}: {len(t)}"
            logger.info(msg) if logger.CURRENT else print(msg)
        return [t]
    
    return load_files(
        paths=paths,
        load_func=load,
        pattern=pattern,
        label_data=label_data,
        throw_error=throw_error
    )


class H5Storage():
    
       
    @classmethod
    def save(
        cls,
        trajectories: Sequence[Trajectory], 
        path: str = '.',
        folder: str = None,
        filename: str = None, 
        extract_info_keys: List[str] = None, 
        compression: str = 'lzf',
        save_separately: bool = True,
        metadata: Optional[Dict] = None,
    ):
        """ Save a sequence of Trajectories to disk using a NumPy-based format.
        
            Args:
            
                trajectories: Sequence of Imitation Trajectory classes
                
                path: Root path for where the folder will be created to save trajectories
                
                folder: Save folder for saving trajectories to
                
                filename: Base name of the save file
                
                extract_info_keys: Keys to be extracted from info and saved as top level 
                    keys.
                    
                compression: Type of compression to be used for h5. See compression
                    types for 
                    
                metadata: Additional metadata to be stored in the attrs of the h5 file.
                
                save_separately: If True, each trajectory will be saved as a separate
                    file instead of a combined file.
                
        """
        folder = "h5" if folder is None else folder
        path = Path(path) / folder
        path.mkdir(parents=True, exist_ok=True)
        
        if save_separately:
            filename =  f"traj" if filename is None else filename
            for i, traj in enumerate(trajectories):
                filename_i =  f"{filename}-{i+1}" 
                cls._save(
                    trajectories=[traj], 
                    path=path / filename_i, 
                    extract_info_keys=extract_info_keys,
                    compression=compression,
                    metadata=metadata
                )
        else:
            filename =  f"trajs-{len(trajectories)}" if filename is None else filename
            cls._save(
                trajectories=trajectories, 
                path=path / filename, 
                extract_info_keys=extract_info_keys,
                compression=compression,
                metadata=metadata
            )
           
     
    @classmethod
    def _save(
        cls,
        trajectories: Sequence[Trajectory],
        path: str = '.',
        extract_info_keys: Dict[str, str] = None, 
        compression: str = 'lzf',
        metadata: Dict = None,
    ):
        """ Core h5 save functionality
        
            Args:
            
                trajectories: Sequence of Imitation Trajectory classes
                
                path: Path trajectories will be saved to.
                
                extract_info_keys: Keys to be extracted from info and saved as top level 
                    keys.
                    
                compression: Type of compression to be used for h5. See compression
                    types for 
                    
                metadata: Additional metadata to be stored in the attrs of the h5 file.
        """
        if not path.name.endswith('.h5'):
            path = path.with_suffix('.h5')
 
        def dataclass_to_h5(dataset, traj, path=''):
            for field in fields(traj):
                class_field = field.name
                if class_field == 'infos':
                    continue
                curr_path = f"{path}.{class_field}" if len(path) > 0 else str(class_field)
                # print(curr_path)
                data = getattr(traj, class_field)
                if class_field == 'attrs':
                    for attr_k, attr_v in data.items():
                        if not isinstance(attr_v, Iterable):
                            attr_v = np.array([attr_v])
                        dataset.attrs[attr_k] = attr_v
                elif is_dataclass(field.type):
                    grp = dataset.create_group(class_field, track_order=True)
                    dataclass_to_h5(grp, data, path=curr_path)
                else:
                    if not isinstance(data, Iterable):
                        data = np.array([data])
                    dataset.create_dataset(
                        f"{class_field}", 
                        data=data, 
                        compression=compression, 
                        track_order=True
                    )
                    
        with h5py.File(path, 'w') as hf:
            for t, traj in enumerate(trajectories):
                print(f"Saving: {len(traj)} to {path}")
                traj_grp = hf.create_group(f"{t}", track_order=True)
                # Store state tuple information
                dataclass_to_h5(traj_grp, traj)
               
                # Extract keys from info to act as part traj top-level data
                # Does not support extracting to sub-group (e.g., feedbacks)
                if extract_info_keys is not None:
                    extracted_info = extract_keys_from_trajectory_info(traj, extract_info_keys)
                    for info_k, info_v in extracted_info.items():
                        traj_grp.create_dataset(
                            f"{info_k}", 
                            data=np.stack(info_v), 
                            compression=compression,
                            track_order=True
                        )
                    
                # Encode and store remaining infos
                info_data = [pickle.dumps(info) for info in traj.infos]
                traj_grp.create_dataset(
                    f"infos", 
                    data=np.stack(info_data).astype('S'), 
                    compression=compression,
                    track_order=True
                )
                
                # Set metadata
                traj_grp.attrs['encoded'] = ['infos']
                traj_grp.attrs['traj_len'] = len(traj)

            hf.attrs['n_trajs'] = len(trajectories)
            hf.attrs['obs_offset'] = np.arange(1, len(trajectories))
            hf.attrs['indices'] = np.cumsum([len(traj) for traj in trajectories[:-1]])
            if metadata is not None:
               for k, v in metadata.items():
                   hf.attrs[k] = v 
                   

    @classmethod
    def load(
        cls,
        path: str,
        traj_class: Union[Trajectory, Callable] = Trajectory,
        use_infos: bool = False,
        keep_info_keys: List[str] = None,
        set_fields: Optional[Mapping[str, str]] = None,
    ):
        assert issubclass(traj_class, Trajectory)
        def decode(info):
            info = pickle.loads(info)
            if keep_info_keys is None:
                return info
            return {k:info[k] for k in keep_info_keys}
        
        trajs = []
        set_fields = {} if set_fields is None else set_fields
        
        def h5_to_dataclass(traj_class, dataset, traj_args=None, path=''):
            traj_args = {} if traj_args is None else traj_args
            for field in fields(traj_class):
                class_field = field.name
                curr_path = f"{path}.{class_field}" if len(path) > 0 else str(class_field)
                h5_field = set_fields.get(curr_path, class_field)
                
                if h5_field == 'attrs':
                    value = dataset.attrs
                elif h5_field == 'infos':
                    if use_infos:
                        value = [decode(info) for info in dataset[h5_field]]
                    else:
                        value = None
                elif h5_field in dataset:
                    value = dataset[h5_field]
                else:
                    return 
                
                if isinstance(value, h5py.Group) and is_dataclass(field.type):
                    grp_args = h5_to_dataclass(field.type, value, path=curr_path) 
                    traj_args[class_field] = field.type(**grp_args)
                elif isinstance(value, h5py.AttributeManager):
                    traj_args[class_field] = {}
                    for k, v in value.items():
                        traj_args[class_field][k] = np.array(v)
                else:
                    traj_args[class_field] = np.array(value) if value is not None else value
            return traj_args
        
        dataset = h5py.File(path, 'r')
        try:
            for t, traj_name in enumerate(dataset.keys()):
                traj_dataset = dataset[traj_name]
                traj_args = h5_to_dataclass(traj_class, traj_dataset)
                trajs.append(traj_class(**traj_args))
                # print(f"Loaded: {len(trajs[0])} from {path}")
        finally:
            dataset.close()

        return trajs
    
    @classmethod
    def append_to_file(cls, data, path):
        """  Appends a new key to a specific group in a h5 file.
        
            Args:
                group:
                
                data:
                
                path:
        """
        def append(key, value, dataset):
            dataset[key] = value
            
        def expand(data, dataset):
            for key, value in data.items():
                if isinstance(value, dict):
                    if key == 'attrs':
                        expand(value, dataset.attrs)
                    elif key not in dataset.keys():
                        dataset.create_group(key, track_order=True)
                        expand(value, dataset[key])
                    else:
                        expand(value, dataset[key])
                else:
                    append(key, value, dataset)
            
        with h5py.File(path, 'a') as dataset:
           expand(data, dataset)
            
    
    
    @classmethod
    def find_group_name(cls, trajectory, path, check_key='obs'):
        """ Given a trajectory, attempts to find corresponding trajectory group in h5 files 
            using check_key for each trajectory. 
        
            Args:
                traj:
                
                path:
        """
        found = []
        with h5py.File(path, 'r') as dataset:
            for h5_name, h5_traj in dataset.items():
                same = np.all(h5_traj[check_key] == getattr(trajectory, check_key))
                if same: found.append(h5_name) 

        if len(found) > 1:
            msg = f"Found multiple group(s) {found} the trajectory could belong to."
            raise ValueError(msg)
        elif len(found) == 0:
            return None
        
        return found[0]
    
class NpzStorage():
    
    @classmethod
    def extract_keys_from_trajectory_info(cls, trajectories, extract_info_keys):
        key_values = {}
        for key in extract_info_keys:
            try:
                values = np.concatenate(
                    [np.stack([info.pop(key) for info in traj.infos]) 
                for traj in trajectories])
                key_values[key] = values
            except KeyError:
                msg = f"Key {key!r} not found in infos. Can not save this key."
                logger.warn(msg) if logger.CURRENT else warnings.warn(msg)
        return key_values

    @classmethod
    def append_to_npz(cls, data, npz_path, overwrite: bool = False):
        """ Appends data to existing npz file
            Reference:
                https://stackoverflow.com/questions/61996146/how-to-append-an-array-to-an-existing-npz-file
        """
        assert os.path.exists(npz_path)
        for name, d in data.items():
            bio = io.BytesIO()
            np.save(bio, np.array(d))
            with zipfile.ZipFile(npz_path, 'a') as zipf:
                file_name = f'{name}.npy'
                if not overwrite and file_name in zipf.NameToInfo.keys():
                    msg = f"File {file_name} already exists. Enable overwrite to overwrite this file. " \
                        "Warning, overwriting can have adverse effects."
                    raise FileExistsError(msg)
                # careful, the file below must be .npy
                zipf.writestr(file_name, data=bio.getbuffer().tobytes())
                
    @classmethod         
    def save(
        cls,
        path: str, 
        trajectories: Sequence[Trajectory], 
        *, 
        extract_info_keys: List[str] = None, 
        metadata: Optional[Dict] = None
    ):
        """ Save a sequence of Trajectories to disk using a NumPy-based format.

            Create an .npz dictionary with the following keys:
            * obs: flattened observations from all trajectories. Note that the leading
            dimension of this array will be `len(trajectories)` longer than the `acts`
            and `infos` arrays, because we always have one more observation than we have
            actions in any trajectory.
            * acts: flattened actions from all trajectories
            * infos: flattened info dicts from all trajectories. Any trajectories with
            no info dict will have their entry in this array set to the empty dictionary.
            * terminal: boolean array indicating whether each trajectory is done.
            * indices: indices indicating where to split the flattened action and infos
            arrays, in order to recover the original trajectories. Will be a 1D array of
            length `len(trajectories)`.

            Args:
                path: Trajectories are saved to this path.

                trajectories: The trajectories to save.
                
                extract_info_keys: Keys to be extracted from info and saved as top level 
                    keys.
                    
                metadata: Additional metadata to be stored with npz file.
                
            Returns:
                A dictionary containing all numpy arrays compressed into npz file.
            
            Raises:
                ValueError: If not all trajectories have the same type, i.e. some are
                    `Trajectory` and others are `TrajectoryWithRew`.
        """
    

        p = parse_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        separate_info_values = cls.extract_keys_from_trajectory_info(
            trajectories=trajectories,
            extract_info_keys=extract_info_keys, 
        ) if extract_info_keys is not None else {}

        infos = [
            # Replace 'None' values for `infos`` with array of empty dicts
            traj.infos if traj.infos is not None else np.full(len(traj), {})
            for traj in trajectories
        ]
        condensed = {
            "obs": np.concatenate([traj.obs for traj in trajectories]),
            "acts": np.concatenate([traj.acts for traj in trajectories]),
            "infos": np.concatenate(infos),
            "terminal": np.array([traj.terminal for traj in trajectories]),
            "indices": np.cumsum([len(traj) for traj in trajectories[:-1]]),
            **separate_info_values
        }
        has_reward = [isinstance(traj, TrajectoryWithRew) for traj in trajectories]
        if all(has_reward):
            condensed["rews"] = np.concatenate(
                [cast(TrajectoryWithRew, traj).rews for traj in trajectories],
            )
        elif any(has_reward):
            raise ValueError("Some trajectories have rewards but not all")

        metadata = {} if metadata is None else metadata
        for name, md in metadata.items():
            condensed[name] = np.array(md)

        np.savez_compressed(path, **condensed)
        
        return condensed
    
    @classmethod    
    def load(
        cls,
        path,
        traj_class: Trajectory = Trajectory,
        use_infos: bool = True, 
        set_fields: Optional[Mapping[str, str]] = None,
    ):  
        set_fields = set_fields if set_fields is not None else {}
        dataset = np.load(path, mmap_mode='r', allow_pickle=True)
        assert 'indices' in dataset
        traj_indices = dataset["indices"]
        num_trajs = len(dataset["indices"])
        # fields = [
        #     # Account for the extra obs in each trajectory as we have 1 more obs than
        #     # action (s, a, s')
        #     np.split(dataset["obs"],  + np.arange(num_trajs) + 1),
        #     np.split(dataset["acts"], traj_indices),
        #     np.split(dataset["infos"], traj_indices) if use_infos else [None]*(num_trajs + 1),
        #     np.split(dataset["terminal"], np.arange(1, num_trajs)),
        # ]
        def npz_to_dataclass(traj_class, path=''):
            fields = []
            field_names = []
            traj_fields = {k:v for k, v in traj_class.__dataclass_fields__.items()}
            for class_field, class_info in traj_fields.items():
                curr_path = f"{path}.{class_field}" if len(path) > 0 else str(class_field)
                npz_field = set_fields.get(curr_path, class_field)
                
                if npz_field not in dataset:
                    continue
              
                # print(curr_path)
                if class_field == 'obs':
                    values = np.split(dataset[npz_field], traj_indices + np.arange(num_trajs) + 1)
                elif class_field == 'infos':
                    values = np.split(dataset[npz_field], traj_indices) if use_infos else [None]*(num_trajs + 1)
                elif class_field == 'terminal':
                    values = np.split(dataset[npz_field], np.arange(num_trajs)+1)
                else:
                    values = np.split(dataset[npz_field], traj_indices) 
                    
                if is_dataclass(class_info.type):
                    f, fn, = npz_to_dataclass(traj_class=class_info.type, path=curr_path)
                    values = [class_info.type(**dict(zip(fn, args))) for args in zip(*f)]
                fields.append(values)
                field_names.append(class_field)
                    
            return fields, field_names
        
        fields, field_names = npz_to_dataclass(traj_class=traj_class)

        # Make sure field_names and fields match or this fails
        # *fields ensures trajectories are grouped together
        return [traj_class(**dict(zip(field_names, args))) for args in zip(*fields)]

    @classmethod
    def accumulate_tags(cls, data, tags):
        assert isinstance(tags, Iterable)
        accum_tags = None
        for d in data:
            for t in tags:
                if t in d:
                    logger.info(f"Found tag {t}, storing in {d}") if logger.CURRENT else ''
                    if accum_tags is None:
                        accum_tags = data[d]
                    else:
                        msg = f"Found tag {d} using {t} does not have same lengths as other tags {len(accum_tags)}"
                        assert len(data[d]) == len(accum_tags), msg
                        accum_tags += data[d]
        return accum_tags
