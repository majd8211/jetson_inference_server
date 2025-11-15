"""
# util_classes.py

Class definitions for post requests, server cache and base wrapper class for
models.
"""

from pydantic import BaseModel
from typing import Dict, Optional, Union

import logging

from config.logging_config import logging_cfg
from gunicorn import glogging
from gunicorn.app.wsgiapp import WSGIApplication

class Cache():
    """
    Store variables and classes that will persist across different requests.
    Can have read/write operations but user must be responsible to avoid
    race conditions.

    NOTE: should be a python DataClass but not available for Python 3.6
    """
    pass

class MethodNotOverrideError(Exception):
    """
    Custom exception class when a method in ModelWrapper has
    not been defined in the inherited class. Raise an error message.
    """

class ModelWrapper():
    """
    A wrapper for all inference models. This base class will define
    what methods must be defined for all inference models. As each model
    will likely require different code for loading weights/inference,
    define the standard method names for override.
    """
    def __init__(
            self, path_to_model_weights: str,
            device: str,
            number_cpu_workers: int,
            batch_size: int
        ) -> None:
        """
        Set required attributes for all inference models.

        :param path_to_model_weights: Absolute path to weights file.
        :type path_to_model_weights: str
        :param device: Whether to use CPU or GPU
        :type device: str
        :param number_cpu_workers: Number of CPU workers for loading data into GPU
        :type number_cpu_workers: int
        :param batch_size: Number of data points per inference.
        """
        self.path_to_model_weights: str = path_to_model_weights
        self.device: str = device
        self.number_cpu_workers: int = number_cpu_workers
        self.batch_size: int = batch_size

        # The actual pytorch/tensor model etc. here ....
        self.model = None
        self.dataloader = None

    def clear_memory(self) -> None:
        """
        Clear the GPU cache.
        NOTE: must be overridden!!
        """
        raise MethodNotOverrideError(
            "Did not override ModelWrapper.clear_memory method!!!"
        )

    def load_model_weights(self) -> None:
        """
        Load the model weights.
        NOTE: must be overridden!!
        """
        raise MethodNotOverrideError(
            "Did not override ModelWrapper.load_model_weights method!!!"
        )

    def inference(self, input_data) -> str:
        """
        Perform inference on the input data ...
        NOTE: must be overridden!!
        """
        raise MethodNotOverrideError(
            "Did not override ModelWrapper.inference method!!!"
        )

class SelectedModel(BaseModel):
    """
    Strict definition of POST request data
    for selecting which model ro run inference on.
    Contains string attribute "name".
    """
    name: str

class CustomGunicornLogger(glogging.Logger):
    """
    Custom logger for Gunicorn log messages.
    """
    def setup(self, cfg):
        """
        Configure Gunicorn application logging.
        """
        logging.config.dictConfig(logging_cfg)

class GunicornApplication(WSGIApplication):
    """
    Class to start gunicorn application from
    Python rather than the commandline.
    """
    def __init__(self, app_uri: str, server_config: Optional[Dict[str,Union[int,str]]]) -> None:
        """
        Class to start gunicorn application from
        Python rather than the commandline.

        :param app_uri: Uri of application to run e.g. main::server
        :type app_uri: str
        :param server_config: Server config.
        :type server_config: Optional[Dict[str,Union[int,str]]]
        """
        self.app_uri = app_uri
        self.server_config = server_config
        super().__init__()

    def load_config(self) -> None:
        """
        Load the config for the base WSGIApplication
        """
        if self.server_config is not None:
            config = {
                key: value
                for key, value in self.server_config.items()
                if key in self.cfg.settings and value is not None
            }
            for key, value in config.items():
                self.cfg.set(key.lower(), value)
