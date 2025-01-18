## Inference for Manacus Male/Female detection in camera trap videos


#### 1. Create a python environment 

- Python 3.9 Installation instructions via Miniconda - https://docs.conda.io/projects/miniconda/en/latest/
- Create a virtual environment named "infer" for this analysis
  ```bash
  conda env remove -n infer
  conda create -n infer python=3.10 -y
  conda activate infer
  ```
- Install the necessary packages
  ```bash
  pip install -r requirements.txt
  ```

#### 2. Configure the inference script
- Ensure that the `*.onnx` file is available locally in the deploy folder.
  ```bash
  wget https://github.com/karmarv/manacus-dynamics/releases/download/v0.2/v02_rtmdet_m_r1_noswitch_allaug_b16_e135.onnx
  ``` 


#### 3. Run inference 
(a.) Image model inference on a sample in deploy folder
```bash
time python rtmdet_infer.py --view-debug --image "./deploy/frame_000830.PNG"
```
- expected output at [./results/frame_000830.PNG.v02.result.jpg](./results/frame_000830.PNG.v02.result.jpg)
  ```log
  2024-12-02 14:05:35.424617 - Process Image: ./deploy/frame_000830.PNG
  Results:  [{'label': 'Female', 'label_id': 1, 'box_xtl': 732, 'box_ytl': 266, 'box_xbr': 843, 'box_ybr': 357, 'confidence': 0.74782014}, 
             {'label': 'Male',   'label_id': 0, 'box_xtl': 1488, 'box_ytl': 318, 'box_xbr': 1560, 'box_ybr': 385, 'confidence': 0.5209318}]
  2024-12-02 14:05:36.027195 - 2 results written to ./results/frame_000830.PNG.v02.result.csv

  real    0m3.045s
  ``` 
(b.) Video model inference on a frame by frame basis for a sample video in deploy folder
```bash
time python rtmdet_infer.py --view-debug --video "./deploy/LM.P4_1.8.22-1.13.22_0127.MP4"
```
- expected output at [./results/LM.P4_1.8.22-1.13.22_0127.MP4.v02.result.csv](./results/LM.P4_1.8.22-1.13.22_0127.MP4.v02.result.csv)
  ```log
  2024-12-02 13:18:00.229832 - Process Video: ./deploy/LM.P4_1.8.22-1.13.22_0127.MP4
  FPS:60.00, (Frames: 1815),       Video:./deploy/LM.P4_1.8.22-1.13.22_0127.MP4 
  100%|██████████████████████████████████████████████████████████████████| 1815/1815 [08:53<00:00,  3.40it/s]
  2024-12-02 13:26:54.095636 - 3362 results written to ./results/LM.P4_1.8.22-1.13.22_0127.MP4.v02.result.mp4

  real    8m56.279s
  ```
