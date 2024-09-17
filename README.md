# Manacus Dynamics 
Camera Trap video processing for Manacus dynamics assessment


## Tasks [Vision]
1. **Dataset Preparation**
    - [ ] Extract video frames given the metadata sheet using [./dataset/fcat/locate_dataset.py](./dataset/fcat/locate_dataset.py)
    - [ ] Organize the dataset for bird detection
      - `Stage 1`: ebird high resolution images in coco format [./dataset/DATASET.md](./dataset/DATASET.md#1-ebird-dataset-preparation)
      - `Stage 2`: camera trap image sampling [TODO]
    - [ ] Optical flow or Background subtraction to determine the most likely frames to contain birds for detection
    - [ ] Run the videos through Marco's audio processing pipeline for likely frames (eg: Bird snaps right after flying off or copulation)
2. **Manacus detection**
    - [ ] Detect bird/manacus in the video frames - `bird`, `background` [Deprecated]
    - [ ] `Stage 1` model to detect manacus in image frames (male/female should be present)
        - male (white chest, black wings) [only male displays aand makes snap noise]
        - female (green) [low representation, camouflages with the foliage]
        - unknown (juvenile: between male and female) [male bird can be practicing infront of juveniles ]
    - [ ] `Stage 2` model trained via transfer learning on camera trap manacus dataset
3. **Dynamics**
    - [ ] Estimate copulation/visitation in the video based on heuristics
        - frame by frame position (bounding box) for male and female


#### Environment Setup

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

#### Dataset
- Refer [./dataset/DATASET.md](./dataset/DATASET.md)

#### Model - Refer [./model/MODEL.md](./model/MODEL.md)
- MMDetection for detection model training
  - Experiments described in [./model/mmdetection/README.DEV.md](./model/mmdetection/README.DEV.md)
- YoloV7 for detection model training
  - Experiments described in [./model/yolov7/README.DEV.md](./model/yolov7/README.DEV.md)
