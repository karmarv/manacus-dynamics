"""
Manacus Recognition Data transformation script

- COCO 1.0 Detection Data annotation format 
- Organize into Train, Val, Test Model Dev sets

"""

import csv
import json
import time
import shutil
import zipfile
import os, glob
import cv2 as cv
import pandas as pd
import datetime


# Defect group labels for deliverable 1A
MANACUS_CLASS_LABELS={
        "Male"      : { "id": 1 , "group": "manacus" }, 
        "Female"    : { "id": 2 , "group": "manacus" },
        "Unknown"   : { "id": 3 , "group": "manacus"  }
}


class FrameExtractor:
    """
    OpenCV utility to sample frames in a video 
    """
    def __init__(self, videopath):
        self.videopath   = videopath    
        self.cap         = cv.VideoCapture(videopath)
        self.fps         = self.cap.get(cv.CAP_PROP_FPS)
        self.width       = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height      = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        #pprint("FPS:{:.2f}, (Frames: {}, Duration {:.2f} s), \t Video:{} ".format(self.fps, self.frame_count, self.frame_count/self.fps, videopath))

    # Extract frame given the identifier
    def image_from_frame(self, frame_id):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
        _, img = self.cap.read()
        return img

def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.now(datetime.timezone.utc).isoformat(" "), 
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
        "id"            : int(image_id),
        "file_name"     : file_name,
        "width"         : image_size[0],
        "height"        : image_size[1],
        "date_captured" : date_captured,
        "license"       : license_id,
        "coco_url"      : coco_url,
        "flickr_url"    : flickr_url
    }
    return image_info


def create_annotation_info(annotation_id, image_id, category_id, area, bounding_box, track_id):
    annotation_info = {
        "id"            : int(annotation_id),
        "image_id"      : int(image_id),
        "category_id"   : int(category_id),   # defect label
        "iscrowd"       : 0,
        "area"          : area,          # float
        "bbox"          : bounding_box,  # [x,y,width,height]
        "segmentation"  : [],            # [polygon]
        "track_id"      : track_id,      # track id based on the start frame id in a track sequence
    }
    return annotation_info


def get_coco_metadata(dataset_type="train"):
    coco_output = {}
    coco_output["info"] = {
        "description": "eBird Manacus Dataset - {}".format(dataset_type),
        "url": "https://github.com/karmarv",
        "version": "0.1.0",
        "year": 2024,
        "contributor": "Rahul Vishwakarma",
        "date_created": datetime.datetime.now(datetime.timezone.utc).isoformat(" "),
    }
    coco_output["licenses"] = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        }
    ]

    coco_output["categories"] = [
        {
            "id": class_label['id'],
            "name": name,
            "supercategory": class_label['group'],
        }
        for name, class_label in MANACUS_CLASS_LABELS.items()
    ]
    coco_output["images"] = []
    coco_output["annotations"] = []
    return coco_output

def read_metadata_file(data_file):
    df = pd.read_csv(data_file)
    print(df.describe())
    return df

def write_json_file(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys = False)

def parse_annotations(df, image_out=None):
    gts_coco_labels = {"images":[], "annotations":[]}
    for index, row in df.iterrows():
        image_idx = str(row['idx_col'])
        image_path = str(row['image'])
        image_label = str(row['sex'])
        frame_ext = FrameExtractor(image_path)
        image_size = (frame_ext.width, frame_ext.height)
        image_info = create_image_info(image_id=image_idx, file_name=os.path.basename(image_path), 
                                           image_size=image_size)
        gts_coco_labels["images"].append(image_info)
        if image_out is not None:
            # copy images to putput path
            output = os.path.join(image_out, os.path.basename(image_path))
            shutil.copy(image_path, output)

        grp_id = MANACUS_CLASS_LABELS[image_label]["id"]
        bb_width, bb_height = int(image_size[0]/3), int(image_size[1]/3)
        area      = bb_width * bb_height
        box       = [bb_width, bb_height, bb_width, bb_height]  # grid image into a third

        # Expecting only 1 bird per image
        ann_info  = create_annotation_info(image_idx, image_idx, grp_id, area, box, 0)
        gts_coco_labels["annotations"].append(ann_info)
    print("Images-{}/Annotations-{} loaded".format(len(gts_coco_labels["images"]), len(gts_coco_labels["annotations"])))
    return gts_coco_labels

def prepare_coco_data():
    image_df = read_metadata_file("./images_all.csv")
    ann_dict = get_coco_metadata("ALL")  
    # organize files in coco dataset format
    coco_labels = parse_annotations(image_df, image_out="coco/images")
    ann_dict["images"]      = ann_dict["images"]      + coco_labels["images"]
    ann_dict["annotations"] = ann_dict["annotations"] + coco_labels["annotations"] 
    write_json_file(os.path.join("coco", 'annotations', 'all_images.json'), ann_dict)

if __name__ == "__main__":
    # annotations and images
    prepare_coco_data()
