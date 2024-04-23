


# Dataset 



### FCAT Dataset Preparation [TODO]
- ![alt text](./fcat/fcat-images-manacus-ctraps.png)
- Download the ["Camera Traps 1 -- Dec 2021 to Jan 2022" >> "Lek 6" dataset](https://tulane.box.com/s/s5qp63p418h7nz4i3tbmcmch6lq2glnx) to a local folder
    - Metadata for the above videos can be found in ["./fcat/spreadsheets/Lek-6_Video-Review_Dec21-Jan22_11.07.23.xlsx"](./fcat/spreadsheets/Lek-6_Video-Review_Dec21-Jan22_11.07.23.xlsx)
    - Verify the video path with [./fcat/locate_dataset.py](./fcat/locate_dataset.py)
        - Creates a filtered file with video availability information
        - Output selected videos to "dataset/videos"
- Extract frames 
    - ffmpeg -i "Lek 6-Pista 1-1.7.22-1.15.22 (incorrect dates on camera)-1.7.22-1.15.22_0047.MP4" -vf "fps=1" frame%04d.png
    - Using [../sample_dataset.py](../sample_dataset.py) based on data sampled in "dataset/videos" with output to "dataset/frames"


### eBird Dataset Preparation

- ![alt text](./ebird/ebird-images-manacus-library.png)
- Download the `Manacus manacus` images dataset from https://support.ebird.org/en/support/solutions/articles/48000838205-download-ebird-data 
- Credit the dataset as per provided guidance at https://support.ebird.org/en/support/solutions/articles/48001064570-crediting-media
- Split the images data for model train, val, test using [./ebird/dataset_split.py](./ebird/dataset_split.py)
- Cocofy the dataset for an initial label with fixed bounding box that can leter be adjusted [./ebird/dataset_cocofy.py](./ebird/dataset_cocofy.py)
- Visualize this dataset using standalone FiftyOne app [./ebird/dataset_visualize.py](./ebird/dataset_visualize.py)
    - ![alt text](./ebird/ebird-dataset-visualize-fiftyone.png)
- Annotate this dataset in CVAT and export labels for training [TODO]
