# Manacus Dynamics 
Camera Trap video processing for Manacus dynamics assessment


### Sub Tasks 
1. Filter dataset that have human labels given the metadata excel sheet
    - Filter on downloaded dataset [./dataset/locate_dataset.py](./dataset/locate_dataset.py)
    - Organize the dataaset for bird detection
2. Run the videos through Marco's audo processing pipeline
    - Bird snaps right after flying off or copulation
3. Detect motion using image processing
    - optical flow in videos
4. Detect bird/manacus in the video frames
    - bird 
    - background
5. Detect manacus display in the video frames (both birds should be present)
    - male (white chest, black wings) [only male displays aand makes snap noise]
    - female (green) [low representation, camouflages with the foliage]
    - juvenile (between male and female) [male bird can be practicing infront of juveniles ]
6. Detect copulation/visitation in the video based on heuristics
    - frame by frame position (bounding box) for male and female



##### Dev Environment

```
conda env remove -n mana
conda create -n mana python=3.9 jupyterlab -c conda-forge
conda activate mana
pip install -r requirements.txt

```