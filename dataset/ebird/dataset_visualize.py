
import os
import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path


"""
Reference: https://docs.voxel51.com/environments/index.html#remote-data

- pip install fiftyone==0.23.8


export FIFTYONE_CVAT_URL=http://karmax:8080/
export FIFTYONE_CVAT_USERNAME=admin
export FIFTYONE_CVAT_PASSWORD=nimda
export FIFTYONE_ANNOTATION_DEFAULT_BACKEND=cvat

"""

# Load COCO data into FiftyOne
# "/home/rahul/workspace/eeb/manacus-project/data-ebird-manacus/coco/"
dataset_dir=  Path().resolve() / "coco" 
name = "test_coco_sample_visualization2"
# The splits to load
splits = ["train", "val", "test"]
# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset
# Import the dataset
dataset = fo.Dataset.from_dir(
    name = name,
    dataset_type=dataset_type,
    data_path=os.path.join(dataset_dir, splits[0], "images"),
    labels_path=os.path.join(dataset_dir, "annotations", "{}.json".format(splits[0])),
    tags=splits[0]
)
#dataset.persistent = True
# View summary info about the dataset
print(dataset)
# Print the first few samples in the dataset
print(dataset.head())
# Visualize
session = fo.launch_app(dataset, remote=True, address="0.0.0.0", port=8085)

selected_view = dataset.view()
# Send the samples to CVAT for annotation
#anno_key = "anno_run_1"
#selected_view.annotate(anno_key, launch_editor=True, classes=["Male","Female","Unknown"], label_type="detections", )
# Annotate in CVAT-
# Load annotations back into dataset
#selected_view.load_annotations(anno_key)


# Blocks execution until the App is closed
session.wait()
