


# Dataset 


### (1.) eBird Dataset Preparation

- ![alt text](./ebird/samples/ebird-images-manacus-library.png)
- Download the `Manacus manacus` images dataset from https://support.ebird.org/en/support/solutions/articles/48000838205-download-ebird-data 
- Credit the dataset as per provided guidance at https://support.ebird.org/en/support/solutions/articles/48001064570-crediting-media
- Split the images data for model train, val, test using [./ebird/dataset_split.py](./ebird/dataset_split.py) [`Deprecated`]
- Cocofy the dataset for an initial label with fixed bounding box that can leter be adjusted [./ebird/dataset_cocofy.py](./ebird/dataset_cocofy.py)
- Visualize this dataset using standalone FiftyOne app [./ebird/dataset_visualize.py](./ebird/dataset_visualize.py)
    - ![alt text](./ebird/samples/ebird-dataset-visualize-fiftyone-v2.png)
- Annotate this dataset in CVAT and export labels for training 
    - Import the cocofied labels into CVAT upon creating task with 3352 images
    - ![alt text](./ebird/samples/ebird-dataset-annotate-cvat.png)
    - [TODO] Label the high resolution images with support from Luke Anderson and experts - http://vader.ece.ucsb.edu:8080/projects/4
    - Split the COCO train, val, test sets using [./ebird/coco/dataset_cocofy.py](./ebird/coco/dataset_cocofy.py)
        - Keeping the category labels balanced 
        ```bash
        $python dataset_cocofy.py
        Total annotations 3372, images 3352
        Copied 2689 train images
        [[   1 1881]
        [   2  788]
        [   3   40]]
        Saved Train 2709 entries in annotations/train.json
        Remaining annotations for val/test split - annotations 3372, images 3352
        Copied 337 val images
        [[  1 238]
        [  2  99]
        [  3   5]]
        Saved Val 342 entries in annotations/val.json
        Copied 337 test images
        [[  1 234]
        [  2 105]
        [  3   6]]
        Saved Test 345 entries in annotations/test.json
        ```
    - Convert COCO to Yolo using [./ebird/yolo/dataset_yolofy.py](./ebird/yolo/dataset_yolofy.py)
        ```bash
        $python dataset_yolofy.py
        Annotations ../coco/annotations/train.json: 100%|████████████| 2689/2689 [00:00<00:00, 44302.32it/s]
        2689it [00:01, 1344.84it/s]
        Annotations ../coco/annotations/val.json: 100%|██████████████| 337/337 [00:00<00:00, 44703.52it/s]
        337it [00:00, 1292.67it/s]
        Annotations ../coco/annotations/test.json: 100%|█████████████| 337/337 [00:00<00:00, 42917.27it/s]
        337it [00:00, 1362.18it/s]
        ``` 

---

### (2.) FCAT Dataset Preparation [TODO]
- Search the Box folder approach
    - ![alt text](./fcat/fcat-images-manacus-ctraps.png)
    - Download the ["Camera Traps 1 -- Dec 2021 to Jan 2022" >> "Lek 6" dataset](https://tulane.box.com/s/s5qp63p418h7nz4i3tbmcmch6lq2glnx) to a local folder
        - Metadata for the above videos can be found in ["./fcat/spreadsheets/Lek-6_Video-Review_Dec21-Jan22_11.07.23.xlsx"](./fcat/spreadsheets/Lek-6_Video-Review_Dec21-Jan22_11.07.23.xlsx)
        - Verify the video path with [./fcat/locate_dataset.py](./fcat/locate_dataset.py)
            - Creates a filtered file with video availability information
            - Output selected videos to "dataset/videos"
    - Extract frames 
        - ffmpeg -i "Lek 6-Pista 1-1.7.22-1.15.22 (incorrect dates on camera)-1.7.22-1.15.22_0047.MP4" -vf "fps=1" frame%04d.png
        - Using [../sample_dataset.py](../sample_dataset.py) based on data sampled in "dataset/videos" with output to "dataset/frames"


- Curated `FemVisitations` videos from SME
    - Spreadsheet (Batch 1) - [./fcat/curated/20240613_FemVisitation_samples.csv](./fcat/curated/20240613_FemVisitation_samples.csv)
    - Sample images (1FPS) with bounding box on objects (>20px) based on optical flow - [Visualization and extraction script]