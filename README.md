# Manacus Dynamics 
Camera Trap video processing for Manacus dynamics assessment


## Tasks [Vision]
1. **Dataset Preparation**
    - [x] Annotate bird/manacus in the video frames with following labels in [CVAT](http://eyeclimate.cnsi.ucsb.edu:8080/projects/5)
        - male (white chest, black wings) [only male displays aand makes snap noise]
        - female (green) [low representation, camouflages with the foliage]
        - unknown (juvenile: between male and female) [male bird can be practicing infront of juveniles]    
    - [x] Organize the dataset for bird detection
      - `Stage 1`: ebird high resolution images in coco format - [DATASET.md#1-ebird-dataset](./dataset/DATASET.md#1-ebird-dataset)
      - `Stage 2`: camera trap image sampling - [DATASET.md#2-fcat-dataset](./dataset/DATASET.md#2-fcat-dataset)
    - Alternate approaches
      - Optical flow or Background subtraction to determine the most likely frames to contain birds for detection
      - Run the videos through Marco's audio processing pipeline for likely frames (eg: Bird snaps right after flying off or copulation)
2. **Manacus detection**
    - [x] `Stage 1` model to detect manacus in eBird image frames (male/female should be present) [YoloV7]
    - [x] `Stage 2` model trained on camera trap manacus dataset [RTMDet]
    - Alternate approaches
      - Pipeline to merge the results of audio and video models
3. **Dynamics**
    - [ ] Estimate copulation/visitation in the video based on heuristics
        - frame by frame position (bounding box) for male and female


### Dataset
- Refer [./dataset/DATASET.md](./dataset/DATASET.md)

### Model 
- Refer [./model/MODEL.md](./model/MODEL.md)
- MMDetection for detection model training
  - Experiments described in [./model/mmdetection/README.DEV.md](./model/mmdetection/README.DEV.md)
  - inference code: [./inference/mm/rtmdet_infer.py](./inference/mm/rtmdet_infer.py)
  - visualizations: https://wandb.ai/karmar/MM-Manacus?nw=nwuserkarmar
- YoloV7 for detection model training
  - Experiments described in [./model/yolov7/README.DEV.md](./model/yolov7/README.DEV.md)
  - inference code: [TODO]()
  - visualizations: https://wandb.ai/karmar/Yv7-Manacus  


### Environment

- Python 3.9 Installation
  - Instructions via Miniconda v23.1.0 - https://docs.conda.io/projects/miniconda/en/latest/
  - Create a virtual environment for this analysis
    ```
    conda env remove -n mana
    conda create -n mana python=3.9 jupyterlab -c conda-forge
    conda activate mana
    ```
- Clone the current codebase - `git clone https://github.com/karmarv/manacus-dynamics.git && cd manacus-dynamics`
- Install pre-requisite packages in the activated python virtual environment using - `pip install -r requirements.txt`