"""
# main.py

Run a fast-api server using Gunicorn (WSGI) with Uvicorn workers (ASGI).
The server will run on Jetson Nano module and will allow model inference via
a POST API request. Since the Jetson Nano has hardware constraints, cannot load
all models into GPU memory at the same time. Hence, there will be another
endpoint where the model to load will be selected, and all inference POST requests
will be handled by that model alone. Going to "/docs" endpoint will return a
Swagger endpoint.
"""

from typing import Dict, List, Optional, Union

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from gunicorn import glogging
from inference import InferenceModel
from util_classes import Cache, CustomGunicornLogger, GunicornApplication, SelectedModel

import uvicorn

load_dotenv()

server_cache: Cache = Cache()

server = FastAPI() # root_path=....

logger = logging.getLogger("app.log")

@server.on_event("startup")
def init_model():
    """
    Load the default model when the
    server is starting.
    """
    logging.debug("Creating variables and classes in cache.")
    server_cache.inferenceModel: InferenceModel = InferenceModel()

@server.get("/")
async def root() -> Dict[str, str]:
    """
    Root end-point will return a welcome message.

    :return: A dictionary with "message" as a key.
    :rtype: Dict[str,str]
    """
    return {"message": "Game on, old friend."}

@server.post("/select_model")
async def select_model(selectedModel: SelectedModel) -> Dict[str, str]:
    """
    Select a model for all future inferences
    to use.

    :param selectedModel: The name of the model to load for later inference.
    :type selectedModel: SelectedModel
    :return: A dictionary with "message" as a key.
    :rtype: Dict[str,str]
    """
    response_str: str = server_cache.inferenceModel.switch_model(selectedModel.name)
    return {"message": f"{response_str}"}

@server.get("/which_model")
async def which_model() -> Dict[str, str]:
    """
    Returns the name of the model being used.

    :return: A dictionary with "message" as a key.
    :rtype: Dict[str,str]
    """
    model_name: str = server_cache.inferenceModel.get_model_name()
    return {"message": f"{model_name}"}

@server.post("/predict_images")
async def predict(image_files: List[UploadFile]) -> Dict[str, Optional[str]]:
    """
    Run inference on one or more images using the selected model.

    :param: image_files: List of images uploaded by the client.
    :type: image_files: List[UploadFile]
    :return: A dictionary with "predicted_data" as a key.
    :rtype: Dict[str, Optional[str]]
    """
    try:
        prediction_results: Optional[str] = await server_cache.inferenceModel.predict_images(
            image_files
        )
    except Exception as e:
        logger.exception(e)
        prediction_results = {"error": str(e)}
    return {"predicted_data": prediction_results}

if __name__ == "__main__":

    server_options: Dict[str, Union[str, int]] = {
        "bind": os.getenv("HOSTNAME") + ":" + os.getenv("PORT"),
        "workers": int(os.getenv("GUNI_WORKERS")),
        "worker_class": "uvicorn.workers.UvicornWorker",
        "logger_class": CustomGunicornLogger
    }
    GunicornApplication("main:server", server_options).run()
