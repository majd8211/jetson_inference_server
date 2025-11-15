"""
# inference.py

Class definition of InferenceModel to load models and perform inference.
"""
from fastapi import UploadFile
from typing import Any, Dict, List, Optional

import importlib
import json
import logging

from util_classes import ModelWrapper
from util_data_load import import_images_tensor

logger = logging.getLogger("app.log")

class InferenceModel():
    """
    Wrapper class for dynamically loading models for ML inference without
    need to restart the service everytime a model is added/removed.
    Also contains class wrapper of the model for inference.
    """
    def __init__(self, path_to_inference_parameters: str = "config/inference_model_parameters.json") -> None:
        """
        Default model to load into GPU when starting up the server.

        :param path_to_inference_parameters: Path of json file containing inference parameters.
        :type path_to_inference_parameters: str
        """
        self.path_to_inference_parameters = path_to_inference_parameters
        self.model_name: str = self.__load_param_json_file__()["default_model_to_load"]
        self.__load_model__()

    def clear_model(self) -> None:
        """
        When changing the model or shutting down the server,
        clear any memory in the GPU.
        """
        if hasattr(self, "inference_model"):
            self.inference_model.clear_memory()
            del self.inference_model
        logger.info("Cleared %s from cache.", self.model_name)

    def get_model_name(self) -> str:
        """
        Getter for the name of the model loaded into the GPU.

        :returns: Name of model loaded into the GPU.
        :rtype: str
        """
        return self.model_name

    def switch_model(self, model_name_to_load: str) -> str:
        """
        Given the name of the model, will search the configuration files
        and load the model into the GPU/CPU, ready for inference.

        :param model_name: The name of the model to load as provided in config files.
        :type model_name: str
        :returns: A message of whether the model was successfully loaded or not.
        :rtype: str
        """
        response_str: str = f"{model_name_to_load} is already loaded"
        if self.model_name != model_name_to_load:
            self.clear_model()
            self.model_name = model_name_to_load
            self.__load_model__()
            response_str = f"{model_name_to_load} is loaded"
            logger.info("Model %s is loaded, ready for inference.", self.model_name)
        return response_str

    async def predict_images(self, image_files: List[UploadFile]) -> Optional[str]:
        """
        Convert the uploaded images from the client into a single tensor array.
        Also extract the original image sizes to rescale the predicted bounding
        box coordinates after inference.

        :param: image_files: List of images uploaded by the client.
        :type: image_files: List[UploadFile]
        :returns: The prediction result of all images.
        :rtype: Optional[str]
        """
        result: Optional[str] = None
        input_images, original_image_sizes = await import_images_tensor(
            image_files, self.inference_model.image_resize,
            self.inference_model.image_number_channels
        )
        result = json.dumps(
            self.inference_model.inference(input_images, original_image_sizes)
        )
        logger.info(
            "Performed inference on images: %s", ",".join(original_image_sizes.keys())
        )
        return result

    def __load_model__(self) -> None:
        """
        Load the model into the GPU.
        Use the config json file which contains the inference arguments.
        Read the json file everytime to ensure latest configurations.
        """

        # Different model weights can have the same model backbone.
        # Account for this case here. E.g. yolov3 vs yolov3-tiny
        model_class_name: str = self.model_name
        if self.model_name == "pytorch_yolov3_tiny" or self.model_name == "pytorch_yolov3_traffic":
            model_class_name = "pytorch_yolov3"

        # Dynamic importing of modules.
        model_wrapper_class: ModelWrapper = importlib.import_module(
                f"model_code.{model_class_name}.model_wrapper", "Model"
            )
        parameters: Dict[str, str] = self.__load_param_json_file__()[self.model_name]
        self.inference_model: ModelWrapper = model_wrapper_class.Model(parameters)
        self.inference_model.load_model_weights()

    def __load_param_json_file__(self) -> Dict[str, Any]:
        """
        Load the json file containing the parameters for model inference
        for all the models and return a dictionary. To ensure that the json
        is human readable by having new lines, need to combine into a single
        string to convert into dictionary.

        :returns: A dictionary of parameters per model. Key is model name.
        :rrtype: Dict[str, Any]
        TODO - check if dictionary has content ...
        """
        parameters: Dict[str, Any] = {}
        with open(self.path_to_inference_parameters) as json_file_in:
            parameters = json.loads(
                "".join(json_file_in.readlines())
            )
        return parameters
