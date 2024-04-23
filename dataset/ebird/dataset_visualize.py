
import os
import fiftyone as fo
import fiftyone.zoo as foz


"""
Reference: https://docs.voxel51.com/environments/index.html#remote-data

- pip install fiftyone==0.23.8
"""

# home\rahul\workspace\eeb\manacus-project\data-ebird-manacus\coco
dataset_dir="/home/rahul/workspace/eeb/manacus-project/data-ebird-manacus/coco/"

name = "test_coco_sample_visualization1"
# The splits to load
splits = ["train", "val", "test"]

# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset  # for example

# Import the dataset
dataset = fo.Dataset.from_dir(
    name = name,
    dataset_type=dataset_type,
    data_path=os.path.join(dataset_dir, splits[0], "images"),
    labels_path=os.path.join(dataset_dir, "annotations", "{}.json".format(splits[0])),
    tags=splits[0]
)

# View summary info about the dataset
print(dataset)

# Print the first few samples in the dataset
print(dataset.head())

# Visualize
session = fo.launch_app(dataset, remote=True, address="0.0.0.0", port=8090)
        
# Blocks execution until the App is closed
session.wait()