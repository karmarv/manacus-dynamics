
# Model

### (1.) Yolov7 - `Stage 1` manacus detection model


```
pip install -r requirements.txt
```



Single GPU training

``` shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
# train p5 models
python train.py --workers 16 --device 0 --batch-size 16 --data data/manacus.yaml --img 640 640 --cfg cfg/training/yolov7-manacus.yaml --weights 'yolov7.pt' --name yv7-manacus --hyp data/hyp.scratch.p5.yaml

```

### (2.) Yolov7 - `Stage 2` Transfer learning camera trap manacus detection model

