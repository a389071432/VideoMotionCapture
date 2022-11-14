## Introduction
**(Please download and read the file 'introduction to project.pdf' for technical details of this project.)**
<br/>
This project gives a solution for an AI-based cloud service, which aims to automatize the production of character animation.

Generally, the project is consisted of three parts :
1. Unity client
   - an interaction interface for users
   - generate and parse a BVH file
   - visualize the motion data in a BVH file 
2. Triton Server
   - providing inference service using deployed neural network models
3. Request Handler
   - communicate with Unity client and Triton
   - data processing
## Use case
Two functionalities are involved : 
- Generate a BVH file<br/>For an input video, a BVH file will be generated which could be directly used in animation editors(e.g., Blender).
  Interpolation and smoothing are supported.
  <p align="center">
   <img src="/demo/inputVideo.gif" /> 
  </p>
  <p align="center">
   <img src="/demo/outputBVH.gif" /> 
  </p>
- Visualize a BVH file<br/>For a loaded BVH file, motion data will be extracted and mapped to the avatar in Unity client, then the animation can play.
  <p align="center">
   <img src="/demo/visualize.gif" /> 
  </p>
## Deployment
- For the frontend, the whole Unity project is involved in the folder```UnityClient```. You just need to open it in Unity. 
- For the backend, two methods of deployment are provided. You can deploy the backend either on a local machine or a remote server. 
## Deploy locally
For a quick use of this project, Triton is not needed, follow : 
1. Install dependencies
<br/>Request Handler runs in a python environment (conda is recommended), make sure following packages are installed properly:
     - numpy
     - pytorch
     - opencv-python, opencv-python-contrib
     - flask
2. Download model weights
<br/>Pre-trained weights are available at https://drive.google.com/drive/folders/1FKDsTK1shHtl1v9i7pB4oGsZGachd1vy?usp=sharing
<br/>put them in the directory ```BackendForLocal/YOLO/weight```, ```BackendForLocal/FastPose/weight``` and ```BackendForLocal/VideoPose3D/weight```, respectively.
3. Run the backend
<br/>simply start the flask server by :
     ```
     python main_local.py
     ```
     Now the bakcend is ready, you can interact with the system through the Unity client provided.
## Deploy with Triton
If you want to run the project as a cloud service using Triton Inference Server, follow :
1. Start a TensorRT environment
<br/>Please follow https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/running.html#running to setup TensorRT. 
<br/>A docker image is highly recommended.
2. ONNX to TensorRT
<br/>ONNX models are available at https://drive.google.com/drive/folders/1Y1RpH2Ild3R4Zz3Pmoc9SlF3lSEukySK?usp=sharing 
<br/>For faster inference, you need to convert the models in 'onnxModels' into .engine files in a TensorRT environment.
<br/>For YOLO, run :
     ```
     trtexec --onnx=yolov3_spp_-1_608_608_dynamic_folded.onnx 
     --explicitBatch 
     --saveEngine=yolov3_spp_-1_608_608_dynamic_folded.engine 
     --workspace=10240 --fp16 --verbose 
     ```
     For FastPose, run :
     ```
     trtexec --onnx=FastPose.onnx 
     --saveEngine=FastPose.engine 
     --workspace=10240 --verbose 
     --minShapes=input:1x3x256x192 
     --optShapes=input:1x3x256x192 
     --maxShapes=input:128x3x256x192 
     --shapes=input:1x3x256x192 
     --explicitBatch
     ```
     Now you get two .engine files, which will be used later. 
     <br/>Note: 
     - VideoPose3D model is not converted, as there is some problem in converting that I haven't figured it out.
     - .engine files are not provided here since the conversion from ONNX to .engine is dependent on what model of GPU you are using.
     
3. Create model repository
<br/>To run on Triton, the three neural network models should be organized in a specific way :

     The folder ```BackendWithTriton/TritonModels``` is ready for being a repository, you just need to rename the obtained engine files as 'model.plan' and put them into ```BackendWithTriton/TritonModels/YOLO```, ```BackendWithTriton/TritonModels/FastPose``` and ```BackendWithTriton/TritonModels/VideoPose3D_onnx```, respectively.
     <br/>
     <br/>```config.pbtxt``` specifies the running configuration of a model. They have been written properly, so you don't need to modify them.
4. Start Triton Inference Server
<br/>Check https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/ and get a released docker image of Triton, make sure that it matches your OS and CUDA. Then, start Triton as a docker container :
     ```
     sudo docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/home/rlt/Desktop/trition:/models nvcr.io/nvidia/tritonserver:<xx.xx>-py3 tritonserver --model-repository=/Models
     ```
     ```/Models``` is the path of the model repository you created in Step3.   
<br/>For detailed description of how to deploy an AI application based on Triton, please refer to the official documentation https://github.com/triton-inference-server/server.
5. Start the Request Handler
<br/>The directory ```BackendWithTriton/RequestHandler``` is actually a simple Flask project, you just need to make it run on the same machine where you deployed Triton for fast communication. You can also extend the RequestHandler as a production-level module using Nginx and Gunicorn.
6. Specify the ip address of server
<br/>After starting the backend on a server machine, you need to specify the ip address of server and the listening port of RequestHandler in the file ```UnityClient/Assets/FileBrowser.cs```, so that the Unity Client can communicate with your backend. By default, the ip and port is set to http://127.0.0.1:8080 for the case of local deployment.  
## Interaction
- In the main interface, you can upload a video and submit it to backend for pose estimation:
<p align="center">
   <img src="/demo/UI_main.png" /> 
</p>

- Estimation results will be sent back to the Unity client, and you need to fill in several settings to create a BVH file:
<p align="center">
   <img src="/demo/UI_settings.png" /> 
</p>

- In the visualization interface, you can select a local BVH file created by this system to load, and it wil be used to create a character animation.
  Several operations are provided, including the buttons for playing, pausing, speed adjustment, as well as a slider to control the animation's progress.
  <p align="center">
   <img src="/demo/UI_visual.png" /> 
  </p>
## Credits
About the three deep learning models used in the backend:
- yolov3-spp+FastPose. Please refer to https://github.com/MVIG-SJTU/AlphaPose
- VideoPose3D. Please refer to https://github.com/facebookresearch/VideoPose3D

