# Manacus Dynamics 
Camera Trap video processing for Manacus dynamics assessment


### Sub Tasks [Vision]
1. **Dataset Preparation**
    - [ ] Filter human label videos given the metadata sheet using [./dataset/locate_dataset.py](./dataset/locate_dataset.py)
    - [ ] Organize the dataset for bird detection
    - [ ] Run the videos through Marco's audio processing pipeline
        - Bird snaps right after flying off or copulation
    - [ ] Detect motion using image processing aand sample frames with bird in it
        - optical flow in videos
2. **Manacus detection**
    - [ ] Detect bird/manacus in the video frames - `bird`, `background`
    - [ ] Detect manacus in image frames (male/female should be present)
        - male (white chest, black wings) [only male displays aand makes snap noise]
        - female (green) [low representation, camouflages with the foliage]
        - juvenile (between male and female) [male bird can be practicing infront of juveniles ]
3. **Dynamics**
    - [ ] Estimate copulation/visitation in the video based on heuristics
        - frame by frame position (bounding box) for male and female


#### Environment Setup

- Python 3.8.18 Installation
  - Instructions via Miniconda v23.1.0 - https://docs.conda.io/projects/miniconda/en/latest/
  - Create a virtual environment named "mothra" for this analysis
    ```
    conda env remove -n mana
    conda create -n mana python=3.9 jupyterlab -c conda-forge
    conda activate mana
    pip install -r requirements.txt
    ```
  - Clone the current codebase - `git clone https://github.com/karmarv/manacus-dynamics.git && cd manacus-dynamics`
  - Install pre-requisite packages in the activated python virtual environment using - `pip install -r requirements.txt`

#### Dataset
- Refer [./dataset/DATASET.md](./dataset/DATASET.md)