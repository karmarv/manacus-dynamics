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
  - v0.1: `wget https://github.com/karmarv/manacus-dynamics/releases/download/v0.1/rtmdet_s_b16_e100.onnx`
  - v0.2: `wget https://github.com/karmarv/manacus-dynamics/releases/download/v0.2/` 


#### 3. Run inference 
(a.) Video model inference on a frame by frame basis for a sample video in deploy folder
```bash
time python rtmdet_infer.py --view-debug --image "./deploy/frame_000830.PNG"
```
- expected output at [./results/frame_000830.PNG.manacus.result.jpg](./results/frame_000830.PNG.manacus.result.jpg)
  ```log
  2024-07-25 17:31:58.425227 - Process Image: ./deploy/frame_000830.PNG
  Results:  [{'label': 'Male', 'label_id': 0, 'points': [1503, 334, 1565, 386], 'type': 'rectangle', 'confidence': '0.7283878'}, 
             {'label': 'Female', 'label_id': 1, 'points': [733, 268, 836, 356], 'type': 'rectangle', 'confidence': '0.7020108'}]
  2024-07-25 17:31:58.554099 - 2 results written to ./results/frame_000830.PNG.manacus.result.csv
  ``` 
(b.) Image model inference on a sample in deploy folder
```bash
time python rtmdet_infer.py --view-debug --video "./deploy/LM.P4_1.8.22-1.13.22_0127.MP4"
```
- expected output at [./results/LM.P4_1.8.22-1.13.22_0127.MP4.manacus.result.mp4](./results/LM.P4_1.8.22-1.13.22_0127.MP4.manacus.result.mp4)
  ```log
  2024-07-25 17:32:19.467363 - Process Video: ./deploy/LM.P4_1.8.22-1.13.22_0127.MP4
  FPS:60.00, (Frames: 1815),       Video:./deploy/LM.P4_1.8.22-1.13.22_0127.MP4
  100%|████████████████████████████████████████████████████████████████████████████| 1815/1815 [03:48<00:00,  7.94it/s]
  2024-07-25 17:36:08.226598 - 1209 results written to ./results/LM.P4_1.8.22-1.13.22_0127.MP4.manacus.result.mp4
  ```
