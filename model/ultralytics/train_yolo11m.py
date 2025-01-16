import os
import cv2
import supervision as sv

import wandb
from wandb.integration.ultralytics import add_wandb_callback

import ultralytics
from ultralytics import YOLO
import torch

ultralytics.checks()

os.environ['OMP_NUM_THREADS'] = '4'  # Adjust this as necessary for your machine
torch.backends.cudnn.benchmark = False
torch.cuda.synchronize()

# Initialize YOLO Model
model = YOLO("yolo11m.pt")
run_name="y11m-dv5-r1"

# Add W&B callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Train the model
with wandb.init() as run:
    train_results = model.train(
        data= 'coco8-fcat-v5.yaml',   # Path to your dataset config file
        epochs= 10,                   # Number of training epochs
        batch = 16,                   # Training batch size
        imgsz= 640,                   # Input image size
        optimizer= 'SGD',             # Optimizer, can be 'Adam', 'SGD', etc.
        lr0= 0.01,                    # Initial learning rate
        lrf= 0.1,                     # Final learning rate factor
        weight_decay= 0.0005,         # Weight decay for regularization
        momentum= 0.937,              # Momentum (SGD-specific)
        verbose= True,                # Verbose output
        device= '3',                  # GPU device index or 'cpu'
        workers= 8,                   # Number of workers for data loading
        project= 'ul-yolo',           # Output directory for results
        name= run_name,               # Experiment name
        exist_ok= False,              # Overwrite existing project/name directory
        rect= False,                  # Use rectangular training (speed optimization)
        resume= False,                # Resume training from the last checkpoint
        multi_scale= False,           # Use multi-scale training
        single_cls= False,             # Treat data as single-class
        cache= False,
        val= True
        #freeze = 20, #default değer : none
        #resume=True, #Başka bilgisatarda eğitim devamı yapılamıyor.
    )
    print("Train Complete: ", train_results)
    # Evaluate model performance on the validation set
    metrics = model.val()
    print("Metrics: ", metrics)

# Perform object detection on an image using the model
sample_results = model("samples/LM.P4_1.8.22-1.13.22_0127.MP4-00000144.jpg", save=True)
print("Sample: ", sample_results)

# Export the model to ONNX format
export_path = model.export(format="onnx")  # return path to exported model
print("Export ONNX: ", export_path)

# Finish the W&B run
wandb.finish()

