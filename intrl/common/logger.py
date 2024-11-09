import sys
import os
import pathlib
import datetime
import tempfile
from typing import Sequence, Optional, Any, List, Union, Callable
from os.path import join
from warnings import warn
from pdb import set_trace

import stable_baselines3.common.logger as sb_logger
from imitation.util import logger as imi_logger
from imitation.util import util
from imitation.data import types
from hydra.experimental.callback import Callback
from hydra.types import TaskFunction

def __make_output_format(
    _format: str,
    log_dir: str,
    log_suffix: str = "",
    max_length: int = 50,
) -> sb_logger.KVWriter:
    """Returns a logger for the requested format.

    Args:
        _format: the requested format to log to
            ('stdout', 'log', 'json' or 'csv' or 'tensorboard').
        log_dir: the logging directory.
        log_suffix: the suffix for the log file.
        max_length: the maximum length beyond which the keys get truncated.

    Returns:
        the logger.
    """

    if _format == "stdout":
        return sb_logger.HumanOutputFormat(sys.stdout, max_length=max_length)
    
    os.makedirs(log_dir, exist_ok=True)
    if _format == "log":
        
        return sb_logger.HumanOutputFormat(
            os.path.join(log_dir, f"log{log_suffix}.txt"),
            max_length=max_length,
        )
    else:
        return sb_logger.make_output_format(_format, log_dir, log_suffix)

def __build_output_formats(
    folder: pathlib.Path,
    format_strs: Sequence[str],
    log_suffix: Optional[str] = '',
) -> Sequence[sb_logger.KVWriter]:
    """ Build output formats for initializing a Stable Baselines Logger. 
    
        This is a reworked version of imitations.utils.logger._build_output_formats
        function. This function now allows for log_suffix to be passed.

        Args:
            folder: Path to directory that logs are written to.
            format_strs: A list of output format strings. For details on available
                output formats see `stable_baselines3.logger.make_output_format`.

        Returns:
            A list of output formats, one corresponding to each `format_strs`.
    """
    log_suffix = '' if log_suffix is None else log_suffix

    if not (len(format_strs) == 1 and 'stdout' in format_strs):
        folder.mkdir(parents=True, exist_ok=True)
    output_formats: List[sb_logger.KVWriter] = []
    for f in format_strs:
        if f == "wandb":
            output_formats.append(imi_logger.WandbOutputFormat())
        else:
            output_formats.append(__make_output_format(f, str(folder), log_suffix))
    return output_formats

def __configure(
    folder: Optional[types.AnyPath] = None,
    format_strs: Optional[Sequence[str]] = None,
    format_strs_suffix: Optional[str] = None,
    
) -> imi_logger.HierarchicalLogger:
    """ Configure Stable Baselines logger to be `accumulate_means()`-compatible.
        
        This is a reworked version of imitations.utils.logger.configure
        function. This function now allows for log_suffix to be passed.
        
        After this function is called, `stable_baselines3.logger.{configure,reset}()`
        are replaced with stubs that raise RuntimeError.

        Args:
            folder: Argument from `stable_baselines3.logger.configure`.
            format_strs: An list of output format strings. For details on available
                output formats see `stable_baselines3.logger.make_output_format`.

        Returns:
            The configured HierarchicalLogger instance.
    """
    if folder is None:
        tempdir = util.parse_path(tempfile.gettempdir())
        now = datetime.datetime.now()
        timestamp = now.strftime("imitation-%Y-%m-%d-%H-%M-%S-%f")
        folder = tempdir / timestamp
    else:
        folder = util.parse_path(folder)
    output_formats = __build_output_formats(folder, format_strs, format_strs_suffix)
    default_logger = sb_logger.Logger(str(folder), list(output_formats))
    hier_format_strs = [f for f in format_strs if f != "wandb"]
    hier_logger = imi_logger.HierarchicalLogger(default_logger, hier_format_strs)
    return hier_logger

class ShadowLogger():
    """ Global logger that shadows Imitation and SB3 loggers """
    CURRENT = None
    LOGGERS = {}
    
    def __getattr__(self, name):
        if hasattr(self.CURRENT, name):
            return getattr(self.CURRENT, name)
        else:
            try:
                self.__dict__[name]
            except KeyError:
                msg = f"Niether {type(self).__name__!r} or {type(self.CURRENT).__name__!r} " \
                      f"objects have no attribute {name!r}"
                raise AttributeError(msg)

    def __setattr__(self, name, value):
        if hasattr(self.CURRENT, name):
            class_name = self.CURRENT.__class__.__name__
            msg = f"You are shadowing the attribute `{name}` which already exists in {class_name}. " \
                  f"Use `self.CURRENT.{name}` to avoid this."
            warn(msg)
        super().__setattr__(name, value)
        
logger = ShadowLogger()

def clear():
    ShadowLogger.CURRENT = None
    ShadowLogger.LOGGERS = {}

def configure(
    folder: Optional[str] = None, 
    format_strs: Optional[Sequence[str]] = None,
    format_strs_suffix: Optional[Sequence[str]] = '',
    set_default: Optional[bool] = True,
    replace: bool = False,
):
    """ Generates and sets global logger using Imitation's logger 
    
        configure() returns a modified stabe_baselines3 logger class based
        on the Imitation's logger.
    """
    logger = __configure(
        folder=folder, 
        format_strs=format_strs, 
        format_strs_suffix=format_strs_suffix
    )
    if set_default:
        ShadowLogger.CURRENT = logger
    add_logger(folder, logger, replace=replace)
    return logger

def get_current_logger():
    """ Quick access to logger being wrapper/shadowed """ 
    return logger.CURRENT

def spawn_child(
    folder:str, 
    root: Optional[str] = None, 
    format_strs: Optional[Sequence[str]]=None,
    format_strs_suffix: Optional[Sequence[str]] = None,
    replace: bool = False,
):
    """ Creates a child logger by reusing aspects of the default logger """
    root = root or logger.dir
    format_strs = format_strs or logger.format_strs
    child = configure(
        folder=join(root, folder), 
        format_strs=format_strs,
        format_strs_suffix=format_strs_suffix,
        set_default=False,
        replace=replace,
    )
    return child

def add_logger(name, logger, replace: bool = False):
    if name in ShadowLogger.LOGGERS and not replace:
        msg = f'Logger {name!r} already exists.'
        raise KeyError(msg)
    
    ShadowLogger.LOGGERS[name] = logger
    
def get_logger(name):
    return ShadowLogger.LOGGERS[name]


class HydraLoggerInit(Callback):
    """ Hydra Callback for building logger on job start 
    
        This allows for each multi-run job to have its own logger.
    """
    def __init__(
        self, 
        folder: Optional[Union[str, List]] = None, 
        format_strs: Optional[Sequence[str]] = None,
        format_strs_suffix: Optional[str] = ''
    ):  
        self.folder = folder
        self.format_strs = format_strs
        self.format_strs_suffix = format_strs_suffix

    def on_job_start(self, config, *, task_function: TaskFunction, **kwargs: Any):
        clear()
        folder = self.folder if isinstance(self.folder, str) else self._build_folder(self.folder)
        configure(
            folder=folder,
            format_strs=self.format_strs, 
            format_strs_suffix=self.format_strs_suffix, 
            set_default=True
        )
    
    def _build_folder(self, folder: List[Union[str, Callable]]):
        return join(*[str(f())if isinstance(f, Callable) else str(f) for f in folder])