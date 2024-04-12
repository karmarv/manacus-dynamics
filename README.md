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


##### Environment Setup

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


### Dataset Preparation
- Download the ["Camera Traps 1 -- Dec 2021 to Jan 2022" >> "Lek 6" dataset](https://tulane.box.com/s/s5qp63p418h7nz4i3tbmcmch6lq2glnx) to a local folder
    - Metadata for the above videos can be found in ["./dataset/spreadsheets/Lek-6_Video-Review_Dec21-Jan22_11.07.23.xlsx"](./dataset/spreadsheets/Lek-6_Video-Review_Dec21-Jan22_11.07.23.xlsx)
    - Verify the video path with [./dataset/locate_dataset.py](./dataset/locate_dataset.py)
        - Creates a filtered file with video availability information
        - Output selected videos to "dataset/videos"
- Extract frames 
    - ffmpeg -i "Lek 6-Pista 1-1.7.22-1.15.22 (incorrect dates on camera)-1.7.22-1.15.22_0047.MP4" -vf "fps=1" frame%04d.png
    - Using [./sample_dataset.py](./sample_dataset.py) based on data sampled in "dataset/videos" with output to "dataset/frames"