"""
Utility functions when transforming input data into the correct
data type and structure for inference.
"""

import numpy as np
import torch

from fastapi import UploadFile
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple

async def import_images_tensor(
        image_files: List[UploadFile],
        image_resize: int,
        number_channels: int = 3
    ) -> [List[torch.FloatTensor], Dict[str, List[int]]]:
    """
    Convert the uploaded list of images for prediction into
    a tensor array of float values for model inference.
    Also resize the images here as not importing the images
    from a folder. Store the original image sizes so that bounding boxes
    from prediction can correctly be rescaled.

    :param image_files: List of images to predict on.
    :type image_files: List[UploadFile]
    :param image_resize: The new height and width of the input images. Must be square.
    :type image_resize: int
    :param number_channels: Number of channels per image. E.g. RGB is 3.
    :type number_channels: int
    :returns: Both the tensor array of input images and dictionary of original image sizes.
    :rrtype: [List[torch.FloatTensor], Dict[str, List[int]]]
    """

    # Cannot append to a numpy array, must allocate memory beforehand.
    image_array = np.empty(
        (len(image_files),image_resize,image_resize,number_channels),
        dtype=np.float32
    )
    original_image_sizes: Dict[str, List[int]] = {} # [width, height]

    for index, input_image in enumerate(image_files):
        image = Image.open(BytesIO(await input_image.read())).convert('RGB')
        original_image_sizes[input_image.filename] = list(image.size)
        image.load()
        image = image.resize((image_resize, image_resize))
        image = np.asarray(image,dtype="float32")/255
        image_array[index] = image

    input_images: List[torch.FloatTensor] = torch.from_numpy(image_array)
    input_images = input_images.permute(0,3,1,2)

    return input_images, original_image_sizes

class InputImageData(Dataset):
    """
    Custom input class to be passed into the dataloader.
    Required as images already in memory, not in a folder.
    """
    def __init__(self, input_images: List[torch.FloatTensor], transform: Any=None) -> None:
        """
        Set the data source to the tensor and transformers.

        :param input_images: Tensor array of input images.
        :type input_images: List[torch.FloatTensor]
        :param transform: Any transformations to run on an input image.
        :type transform: Any
        """
        self.input_images = input_images
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of images to pass into the model.

        :returns: Length of images for prediction.
        :rtype: int
        """
        return len(self.input_images)

    def __getitem__(self, idx: int) -> torch.FloatTensor:
        """
        Executed by the dataloader to return transformed images
        as a tensor object.

        :param idx: The index of the image to return.
        :type: idx: int
        :returns: The image to run prediction on.
        :rtype: torch.FloatTensor
        """
        images_to_predict: torch.FloatTensor = self.input_images[idx]

        if self.transform is not None:
            images_to_predict = self.transform(images_to_predict)
        return images_to_predict

def compute_mean_std_per_channel_RGB(input_images: List[torch.FloatTensor]) -> List[Tuple[float]]:
    """
    Calculate the mean and standard deviation of each channel for RGB images.
    Could be useful for normalisation the images in transformation prior to
    inference. Result will be:

        [(mean1, mean2, mean3), (std1, std2, std3)]

    :param input_images: Tensor array of input images.
    :type input_images: List[torch.FloatTensor]
    :returns: Mean and standard deviation for each channel.
    :rtype: List[Tuple[float]]
    """

    mean0 = input_images[:,0,:,:].mean()
    mean1 = input_images[:,1,:,:].mean()
    mean2 = input_images[:,2,:,:].mean()

    std0 = input_images[:,0,:,:].std()
    std1 = input_images[:,1,:,:].std()
    std2 = input_images[:,2,:,:].std()

    return (mean0,mean1,mean2),(std0,std1,std2)
