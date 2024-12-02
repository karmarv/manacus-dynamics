import os
import json
import argparse






"""
Get the image id and annotation sequence id for next dataset
"""
def get_max_ids_annotations(coco):
    max_image_id = 0
    for entry in coco["images"]:
        if int(entry['id'])>max_image_id:
            max_image_id = int(entry['id'])
    max_anns_id = 0
    for entry in coco["annotations"]:
        if int(entry['id'])>max_anns_id:
            max_anns_id = int(entry['id'])
    return max_image_id+1, max_anns_id+1

def get_coco_base_annotations(coco_json):
    with open(coco_json) as annf:
        coco_extend = json.load(annf)
        len_images, len_labels = get_max_ids_annotations(coco_extend)
    return coco_extend, len_images, len_labels



"""
Usage: 
- time python dataset_coco_merge.py  --output-dir ./coco/fcat-manacus-v5-ebplus/ --coco-ext ./coco/fcat-manacus-v4-inter/annotations/train.json

Annotations:
- categories: Stores the class names for the various object types in the dataset. Note that this toy dataset only has one object type.
- images: Stores the dimensions and file names for each image.
- annotations: Stores the image IDs, category IDs, and the bounding box annotations in [Top-Left X, Top-Left Y, Width, Height] format.
"""
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Convert SVRDD annotations to ImageNet dataset')
    parser.add_argument("--output-dir",    type=str,  default="./coco/fcat-manacus-v5-ebplus/",  help="Output directory for final COCO dataset")
    parser.add_argument("--coco-ext",    type=str,  default="./coco/fcat-manacus-v4-inter/annotations/train.json",  help="Extend and append new data to existing coco dataset json file")
    args=parser.parse_args()

    in_coco_train = args.coco_ext