import os
import json
import argparse
import datetime
from tqdm import tqdm

from ineye import viz

# labels for Manacus 
MANACUS_CLASS_LABELS={
        "Male"      : { "id": 1 , "group": "manacus" }, 
        "Female"    : { "id": 2 , "group": "manacus" },
        "Unknown"   : { "id": 3 , "group": "manacus"  }
}

"""
Get the image id and annotation sequence id for next dataset
"""
def get_max_ids_annotations(coco):
    max_image_id = 0
    min_image_id = int(coco["images"][0]['id'])
    for img in coco["images"]:
        if int(img['id'])>=max_image_id:
            max_image_id = int(img['id'])
        if int(img['id'])<min_image_id:
            min_image_id = int(img['id'])
    max_anns_id = 0
    min_anns_id = int(coco["annotations"][0]['id'])
    for ann in coco["annotations"]:
        if int(ann['id'])>=max_anns_id:
            max_anns_id = int(ann['id'])
        if int(ann['id'])<min_anns_id:
            min_anns_id = int(ann['id'])
    return max_image_id+1, max_anns_id+1, min_image_id, min_anns_id

def read_coco_annotations(coco_json):
    with open(coco_json) as annf:
        coco_extend = json.load(annf)
        max_image_id, max_anns_id, min_image_id, min_anns_id = get_max_ids_annotations(coco_extend)
    return coco_extend, max_image_id, max_anns_id, min_image_id, min_anns_id


def get_coco_metadata(dataset_type="train"):
    coco_output = {}
    coco_output["info"] = {
        "description": "Manacus COCO eBird+FCAT Dataset - {}".format(dataset_type),
        "url": "https://github.com/",
        "version": "0.2.0",
        "year": 2024,
        "contributor": "Rahul Vishwakarma",
        "date_created": datetime.datetime.utcnow().isoformat(" "),
    }
    coco_output["licenses"] = [
        {
            "id": 1,
            "name": "[TODO] Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        }
    ]
    coco_output["categories"] = [
        {
            "id": class_id['id'],
            "name": class_name,
            "supercategory": class_id['group'],
        }
        for class_name, class_id in MANACUS_CLASS_LABELS.items()
    ]
    coco_output["images"] = []
    coco_output["annotations"] = []
    return coco_output

def update_coco_annotations(coco_extn, img_id_max, ann_id_max):
    """Incremnent the id values for image and corresponding annotation entries"""
    images, annots = [], []
    # Update image id with max offset value
    for img in coco_extn["images"]:
        img["id"] = img["id"] + img_id_max
        images.append(img)
    # Update annotation id with max offset value
    for ann in coco_extn["annotations"]:
        ann["id"] = ann["id"] + ann_id_max
        ann["image_id"] = ann["image_id"] + img_id_max
        annots.append(ann)
    return images, annots

def verify_annotation_images(ann_file, images, annots, output_dir):
    """ Verify valid images in annotations """
    split_type = os.path.basename(ann_file).split('.')[0]
    images_dict = {}
    img_notfound, img_count = 0, 0
    for idx, img in enumerate(images):
        img_path = os.path.join(output_dir, '..', split_type, 'images', img["file_name"])
        #if idx < 3:
        #    print(img_path)
        if os.path.isfile(img_path):
            images_dict[img["id"]] = img
            img_count = img_count + 1
        else:
            img_notfound = img_notfound + 1
    if img_count != len(images):
        print("\t>Images not found {}".format(img_notfound))

    # verify if image id is in annotations
    ann_miscount, ann_count = 0, 0
    for ann in annots:
        ann["id"] = ann["id"] + ann_id_max
        if ann["image_id"] not in images_dict.keys():
            ann_miscount = ann_miscount + 1
        else:
            ann_count = ann_count + 1
    if ann_count != len(annots):
        print("\t>Mismatched annotations {}".format(ann_miscount))
    return ann_miscount, ann_count

"""
Annotations:
- categories: Stores the class names for the various object types in the dataset. Note that this toy dataset only has one object type.
- images: Stores the dimensions and file names for each image.
- annotations: Stores the image IDs, category IDs, and the bounding box annotations in [Top-Left X, Top-Left Y, Width, Height] format.

Images: Copy all the test, val, train images folders to the target output path
- cp ../ebird/coco/train/images/* ./coco/fcat-manacus-v5-fcat-ebird/train/images/
- cp ../ebird/coco/test/images/* ./coco/fcat-manacus-v5-fcat-ebird/test/images/
- cp ../ebird/coco/val/images/* ./coco/fcat-manacus-v5-fcat-ebird/val/images/

Usage: Create the train, val and test JSON and verify if all the images were available as per COCO structure

- time python dataset_coco_merge.py 
    # Sample output
    Namespace(output_dir='./coco/fcat-manacus-v5-fcat-ebird/annotations/', coco_base='../ebird/coco/annotations/', coco_extn='./coco/fcat-manacus-v4-inter/annotations/')

    Read BASE train.json,   max(img_id:3353, ann_id:3373), min(img_id:1, ann_id:1)
    Read BASE val.json,     max(img_id:1336, ann_id:1347), min(img_id:2, ann_id:2)
    Read BASE test.json,    max(img_id:1338, ann_id:1349), min(img_id:2, ann_id:2)
        > img_id_max: 3353, ann_id_max: 3373 
    Read EXTN train.json, max(img_id:192094, ann_id:260894), min(img_id:1, ann_id:1)
        > Extend Images (153674), Annotations (208681)
        > Combined Images (156363), Annotations (211390) written to ./coco/fcat-manacus-v5-fcat-ebird/annotations/train.json
    Read EXTN val.json, max(img_id:192093, ann_id:260893), min(img_id:8, ann_id:8)
        > Extend Images (19209), Annotations (26106)
        > Combined Images (19546), Annotations (26448) written to ./coco/fcat-manacus-v5-fcat-ebird/annotations/val.json
    Read EXTN test.json, max(img_id:192084, ann_id:260884), min(img_id:15, ann_id:15)
        > Extend Images (19210), Annotations (26106)
        > Combined Images (19547), Annotations (26451) written to ./coco/fcat-manacus-v5-fcat-ebird/annotations/test.json
    real    0m11.051s
"""
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Merge train.json, val.json, test.json files in two separate COCO dataset based on the image & annotation identifier sequence')
    parser.add_argument("--output-dir",  type=str,  default="./coco/fcat-manacus-v5-fcat-ebird/annotations/",        help="Output directory for final COCO dataset")
    parser.add_argument("--coco-base",   type=str,  default="../ebird/coco/annotations/",                       help="Base coco annotations json files")
    parser.add_argument("--coco-extn",   type=str,  default="./coco/fcat-manacus-v4-inter/annotations/",   help="Addition to base coco dataset json files")
    args=parser.parse_args()
    print("{}\n".format(args))
    # Read base coco annotations
    ann_file_splits = ["train.json", "val.json", "test.json"]
    ann_base_splits = {}
    img_id_max, ann_id_max = 0, 0
    img_id_min, ann_id_min = 0, 0
    
    for ann_file in ann_file_splits:
        cc_base, b_img_id_max, b_ann_id_max, b_img_id_min, b_ann_id_min = read_coco_annotations(args.coco_base+ann_file)
        ann_base_splits[ann_file] = cc_base
        if b_img_id_max >= img_id_max:
            img_id_max = b_img_id_max
        if b_ann_id_max >= ann_id_max:
            ann_id_max = b_ann_id_max
        print("Read BASE {}, max(img_id:{}, ann_id:{}), min(img_id:{}, ann_id:{})".format(ann_file, b_img_id_max, b_ann_id_max, b_img_id_min, b_ann_id_min))
    print("> img_id_max: {}, ann_id_max: {} \n".format(img_id_max, ann_id_max))

    # Read extend coco annotations
    for ann_file in ann_file_splits:
        cc_extn, e_img_id_max, e_ann_id_max, e_img_id_min, e_ann_id_min = read_coco_annotations(args.coco_extn+ann_file)
        print("Read EXTN {}, max(img_id:{}, ann_id:{}), min(img_id:{}, ann_id:{})".format(ann_file, e_img_id_max, e_ann_id_max, e_img_id_min, e_ann_id_min))

        images, annots = update_coco_annotations(cc_extn, img_id_max, ann_id_max)
        print("\t> Extend Images ({}), Annotations ({})".format(len(images), len(annots)))
        # Add the base to extn coco data
        coco_new = get_coco_metadata(dataset_type=ann_file)
        coco_new["images"] = images + ann_base_splits[ann_file]["images"]
        coco_new["annotations"] = annots + ann_base_splits[ann_file]["annotations"]
        miscount, allcount = verify_annotation_images(ann_file, coco_new["images"], coco_new["annotations"], args.output_dir)

        # Write to output
        viz.write_json_file(os.path.join(args.output_dir, ann_file), coco_new)
        print("\t> Combined Images ({}), Annotations ({}) written to {}".format(len(coco_new["images"]), len(coco_new["annotations"]), os.path.join(args.output_dir, ann_file)))
