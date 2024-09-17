
##### Yolov9 - `Stage 1` manacus detection model

Training logs on W&B - https://wandb.ai/karmar/Yv9-Manacus

```bash
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt
# train yolov9 models
python train_dual.py --workers 8 --device 0 --batch 16 --data data/manacus.yaml --img 640 --cfg models/detect/yolov9-c-manacus.yaml --weights 'yolov9-c-converted.pt' --name yv9-c-manacus --hyp hyp.scratch-high.yaml --min-items 0 --epochs 300 --close-mosaic 15
```
- [r1] Yv9-c model trained on SME labeled dataset 
  ```log
  # In-progress
  ```

