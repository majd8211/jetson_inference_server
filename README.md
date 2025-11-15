# jetson_inference_server
Python API server to expose ML and LLMs on Jetson Nano for Inference

This repo is to run a fast-api server using Gunicorn (WSGI) with Uvicorn workers (ASGI). The server will run on Jetson Nano hardware and will allow model inference via a POST API request. Since the Jetson Nano has hardware constraints, you cannot load all models onto GPU at the same time. Hence, there will be another endpoint where the model to load will be selected, and all inference POST requests will be handled by that model alone. Going to "/docs" endpoint will return a Swagger endpoint.

Current list of available models:

- `pytorch_yolov3`: taken from https://github.com/Borda/PyTorch-YOLOv3 on a Darknet-53 backend already pre-trained on ImageNet.
- `pytorch_yolov3_tiny`: smaller than the `pytorch_yolov3` but less accurate in instance segemtation.
- `pytorch_yolov3_traffic`: a custom trained model on the `pytorch_yolov3` architecture with pre-trained weights on ImageNet. 

The `traffic` model is to locate the sign and the type of sign from an image. An application where this will be useful will be an autonomous car.

Each Jetson Nano module will host one api-server. For load-balancing of requests, a custom Nginx podman image was built where you can dynamically add or remove hosts via a separate API. The Dockerfile can be found in the `jetson_gateway_server` repo.

**Note**: reason for the old version three of YOLO is due to the old Ubuntu 20 operating system and the limited 4GB VRAM on the Jetson Nano. As a consequence, older versions of Python, Torch and Nividia drivers were compatiable.

## Setup Server

### Base Install of Packages

First need to install basic torch, Pillow packages etc. Follow the order to avoid dependency issues. Difficult to install OpenCV. It is already part of the Jetpack as `4.1.1-2`. So don't include it inside the requirements file. Therefore, create a symlink:
```
$ virtualenv venv -p /usr/bin/python3.6
$ ln -s /usr/lib/python3.6/dist-packages/cv2 <absolute path to venv>/venv/lib/python3.6/site-packages/cv2
```

Install the following packages that don't require a `.tar` file:
```
$ python -m pip install cython numpy==1.19.4 scipy gitpython ipython requests tqdm tensorboard pandas
```

Install the following packages which require a `.tar` file:
```
PyYAML
docopt
psutil
pyrsistent
Pillow
matplotlib==3.2.2 # Specific version compatiable.
```

Need to install specific `torch` and `torchvision` packages for the Nano.
```
torch-10.0.0a0:git364....aarch64.whl
torchvision-0.11.0a0=fa347....aarch64.whl
```

Now install the remaining packages which require other dependencies:
```
$ python -m pip install seaborn thop
```

### Install Packages for FastAPI Server

```
$ python -m pip install -r requirements_server.txt
```

### Install Remaining Packages for Specific Model

The Python packages for each specific model is in `./model_code/<model_name>/<original_model_folder>/requirements.txt`

## Adding A New Model

- Ensure you have the evaluation weights saved, not the training checkpoints. This will make sure that you can move the weights and the python code into separate directories.
- Create a folder under `model_code` with the name of the model and place the relevant files underneath as another subdirectory.
- Create a `model_wrapper.py` in the new folder and inherit the ModelWrapper class. This will serve as the interface between the standard inferenceModel class and the classes defined within the original code.
- Configure the `inference_model_parameters.json`. Compulsory parameters are `path_to_model_weights`, `device`, `number_cpu_workers` and `batch_size`.
- Under `./resources` directory is where model configuration, labels and weights will be placed.

Best example is `pytorch_yolov3` setup.

## Usage - Client Code

Example client code to send images to gateway for inference:
```
import requests
import json

files_to_send = [
    "inference_images/dog.jpg",
    "inference_images/eagle.jpg",
    "inference_images/field.jpg"
]

for i in range(4):

    files = [("image_files", (
        filename.split("/")[-1], open(filename, "rb"), "image/jpg")
    ) for filename in files_to_send]
    print(files)

    response = requests.post(
        "http://<hostname>:<port>/predict_images",
        files = files
    )

    print(response)
    print(response.json()) # Contains top-left image coordinates and bounding box dimensions + class
```

## Usage - Manual Deploy of Fast-API Server

First, create a `.env` file:
```
HOSTNAME=
PORT=
GUNI_WORKERS=1
```

Command to start the server in the background:
```
$ nohup python main.py &
```

All logs will be sent to Grafana for monitoring.

## TODO:

- Unit tests for API server.
- Fix issue with Gitlab CI/CD Runner unable to build the Nividia Docker image containing the fast-api server.

## Resources:

- Setting up Gunicorn and Nginx: https://dylancastillo.co/fastapi-nginx-gunicorn/
- Gunicorn and Uvicorn: https://fastapi.tiangolo.com/deployment/server-workers/
- FastAPI load balancer Nginx: https://stackoverflow.com/questions/77460041
- Providing root path for fastAPI behind root proxy: https://fastapi.tiangolo.com/advanced/behind-a-proxy/
- Password for FastAPI authentication: https://fastapi.tiangolo.com/advanced/security/http-basic-auth/
- Nginx, Gunicorn and FastAPI: https://docs.vultr.com/how-to-deploy-fastapi-applications-with-gunicorn-and-nginx-on-ubuntu-20-04
- Load balancing with Nginx: http://nginx.org/en/docs/http/load_balancing.html