# Manacus Dynamics 
Camera Trap video processing for Manacus dynamics assessment


### Sub Tasks 
- Filter dataset that have human labels given the metadata excel sheet
- Run the videos through Marco's audo processing pipeline
- Detect motion using optical flow in videos
- Detect bird/manacus in the video frames
- Detect male/female manacus in the video frames 
- Detect copulation/visitation in the video based on heuristics



##### Dev Environment

```
conda env remove -n mana
conda create -n mana python=3.9 jupyterlab -c conda-forge
conda activate mana
pip install -r requirements.txt

```