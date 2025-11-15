"""
Wrapper of torch_yolov3 model from https://github.com/Borda/PyTorch-YOLOv3
Extension of ModelWrapper base class to define the standard methods invoked
by the InferenceModel class.
"""

from typing import Dict, List, Union

import numpy as np
import torch

from model_code.pytorch_yolov3.torch_yolov3.datasets import ImageFolder
from model_code.pytorch_yolov3.torch_yolov3.evaluate import non_max_suppression
from model_code.pytorch_yolov3.torch_yolov3.models import Darknet
from model_code.pytorch_yolov3.torch_yolov3.utils import load_classes, rescale_boxes
from util_classes import ModelWrapper
from util_data_load import InputImageData

from torch.autograd import Variable
from torch.utils.data import DataLoader

class Model(ModelWrapper):
    """
    Wrapper of torch_yolov3 model from https://github.com/Borda/PyTorch-YOLOv3
    """
    def __init__(self, parameters: Dict[str, str]) -> None:
        """
        Set model parameters required for inference with this model.
        """

        # Attributes common across all inference models
        super().__init__(
            parameters["path_to_model_weights"], parameters["device"],
            parameters["number_cpu_workers"], parameters["batch_size"]
        )

        # Attributes specific to only torch_yolov3
        self.path_to_model_config: str = parameters["path_to_model_config"]
        self.path_to_model_labels: str = parameters["path_to_model_labels"]
        self.image_resize: int = parameters["image_resize"]
        self.image_number_channels: int = parameters["image_number_channels"]
        self.confidence_threshold: Float = parameters["confidence_threshold"]
        self.nms_threshold: Float = parameters["nms_threshold"]

        self.tensor = torch.cuda.FloatTensor if self.device == "cuda" else torch.FloatTensor

    def clear_memory(self) -> None:
        """
        Clear the GPU cache.
        """
        del self.dataloader
        torch.cuda.empty_cache()


    def load_model_weights(self) -> None:
        """
        Load the model weights, set to evaluation mode.
        Also load all classes which the model can predict.
        """
        self.model = Darknet(
            self.path_to_model_config, img_size = self.image_resize
        ).to(self.device)
        self.model.load_darknet_weights(self.path_to_model_weights)
        self.model.eval()

        # Loading all classes the model can predict
        self.classes = load_classes(self.path_to_model_labels)


    def inference(self, input_data_tensor: List[torch.FloatTensor],
            input_images_sizes: Dict[str, List[int]]
        ) -> Dict[str, List[Union[str, float]]]:
        """
        Run inference over the images, returning
        the labels, the coordinates of the bounding box
        and prediction confidence.

        :param input_data_tensor: A torch array of the input images.
        :type input_data_tensor: List[torch.FloatTensor]
        :param input_images_sizes: The original [width height] for each input image.
        :type input_images_sizes: Dict[str, List[int]]
        :returns: A dictionary with image path as key, and BB coords and label as value.
        :rrtype: Dict[str, List[Union[str, float]]]
        """

        # Do not shuffle as image order is important when extracting results.
        input_data: InputImageData = InputImageData(input_data_tensor)
        self.dataloader = DataLoader(
            input_data, batch_size = self.batch_size,
            shuffle = False, num_workers = self.number_cpu_workers
        )

        # Inference here with batches of input images.
        img_detections = []
        for input_imgs in self.dataloader:
            input_imgs = Variable(input_imgs.type(self.tensor))
            with torch.no_grad():
                detects = self.model(input_imgs)
                detects = non_max_suppression(
                    detects, self.confidence_threshold, self.nms_threshold
                )
            img_detections.extend(detects)

        # Do not convert results into string as response object is a dictionary
        return self.__extract_results__(img_detections, input_images_sizes)

    def __extract_results__(
            self,
            img_detections: List[torch.FloatTensor],
            input_images_sizes: Dict[str, List[int]],
        ) -> Dict[str, List[Union[str, float]]]:
        """
        Convert box position into ratio of image and detections as a useful label.
        img_detections contains a list of lists, each list is for one image
        and within each list are prediction values of the form (float values):

            [x1, y1, x2, y2, confidence, class_confidence, class_prediction]

        If nothing was predicted on the image, will just return None. The bounding
        boxes will be rescaled to the original dimensions of the input image.

        NOTE: most of this code was stolen from the OG implementation.

        :param img_detections: The predicted results from the Yolo Model per batch.
        :type img_detections: List[torch.FloatTensor]
        :param input_images_sizes: The original [width height] for each input image.
        :type input_images_sizes: Dict[str, List[int]]
        :returns: A dictionary with image path as key, and BB coords and label as value.
        :rrtype: Dict[str, List[Union[str, float]]]
        """
        output_detections: Dict[str, List[Union[str, float]]] = {}
        for image_path, image_detections in zip(input_images_sizes.keys(), img_detections):
            img_height, img_width = input_images_sizes[image_path]
            if image_detections is not None:
                rescale_detections = rescale_boxes(
                    image_detections, self.image_resize, [img_height, img_width]
                )
                raw_detect = []
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in rescale_detections:
                    box_width = float(x2 - x1)
                    box_height = float(y2 - y1)
                    box_centre_x = float((x2 + x1) / 2)
                    box_centre_y = float((y2 + y1) / 2)
                    bbox = [
                        np.round(box_centre_x / img_width, 5), np.round(box_centre_y / img_height, 5),
                        np.round(box_width / img_width, 5), np.round(box_height / img_height, 5),
                        conf.item(), cls_conf.item(), self.classes[int(cls_pred)]
                    ]
                    raw_detect.append(bbox)
            output_detections[image_path] = raw_detect
        return output_detections
