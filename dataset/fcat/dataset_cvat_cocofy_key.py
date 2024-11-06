import csv
import json
import time
import shutil
import zipfile
import os, glob
import argparse
import datetime
import traceback

import cv2 as cv
import pandas as pd
from tqdm import tqdm

# labels for Manacus 
MANACUS_CLASS_LABELS={
        "Male"      : { "id": 1 , "group": "manacus" }, 
        "Female"    : { "id": 2 , "group": "manacus" },
        "Unknown"   : { "id": 3 , "group": "manacus"  }
}

# Counts
COUNT_ID_DICT = {"img_id":0, "ann_id":0, "trk_id":0, "labels":[]}


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
        print("FPS:{:.2f}, (Frames: {}, Duration {:.2f} s), \t Video:{} ".format(self.fps, self.frame_count, self.frame_count/self.fps, videopath))

    # Extract frame given the identifier
    def image_from_frame(self, frame_id):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
        _, img = self.cap.read()
        return img

def get_clean_basename(filename):
    """ Clean alphanumeric base filename string except '._-' """
    if filename is None:
        return None
    basename = os.path.basename(filename)
    clean_basename = "".join( x for x in basename if (x.isalnum() or x in "._-"))
    return clean_basename

def save_key_frames(basename, frame_extractor, frame_ids, frame_step_size, output_dir):
    """
    Extract the frame ids from video 
    """
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    frame_files = []
    for frame_id in frame_ids:
        # Assign filename as per annotations
        framename = "{}-{:08d}.jpg".format(basename, frame_id)
        frame_path = os.path.join(images_dir, framename)
        if frame_step_size>1:
            # Extract the step size based frame from video
            frame_id = frame_id * frame_step_size
        img = frame_extractor.image_from_frame(frame_id)
        # Write frames. Raises exception if image not read.
        cv.imwrite(frame_path, img)
        frame_files.append([os.path.join("images", framename)])

    # Write frame path to file
    list_file = os.path.join(output_dir, 'images.txt')
    write_list_file(list_file, frame_files)
    return 

def copy_extracted_keyframe_ids(basename, frame_ids, src_images_dir, output_dir):
    """
    Copy the frame ids from extracted video images
    """
    dst_images_dir = os.path.join(output_dir, 'images')
    os.makedirs(dst_images_dir, exist_ok=True)

    frame_files = []
    for frame_id in frame_ids:
        # Assign filename as per annotations frame id (stepsize adjusted ids while reading labels)
        framename = "{}-{:08d}.jpg".format(basename, frame_id)
        frame_path = os.path.join(src_images_dir, framename)
        if os.path.isfile(frame_path):
            # Copy frames raises exception if image not read.
            shutil.copy2(src=frame_path, dst=os.path.join(dst_images_dir, framename))
            frame_files.append([os.path.join("images", framename)])
        else:
            msg = "Id: {}, Invalid Frame: {}, ".format(frame_id, frame_path)
            print(msg)
            raise Exception(msg)
    # Write frame path to file
    list_file = os.path.join(output_dir, 'images.txt')
    write_list_file(list_file, frame_files)
    return 



# File helper utilities

"""
Write list to file or append if it exists
"""
def write_list_file(filename, rows, delimiter=','):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a+") as my_csv:
        csvw = csv.writer(my_csv,delimiter=delimiter)
        csvw.writerows(rows)

def write_json_file(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_json(filename, key=None):
    data = {}
    with open(filename) as jf:
        data = json.load(jf)
    if key is not None:
        return data[key]
    else:
        return data

def get_frame_step_size(data):
    frame_step_size=1
    frame_filter = data["frame_filter"]
    frame_filter = frame_filter.split("=")
    if len(frame_filter)>1:
        frame_step_size = int(frame_filter[1])
    return frame_step_size


# COCO helper utilities

def get_attributes(attrs, key):
    """
    Get value of class label attributes
    """
    for item in attrs:
        if item['name'] == key:
            return item['value']
    return None

def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(" "), 
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
        "id"            : image_id,
        "file_name"     : file_name,
        "width"         : image_size[0],
        "height"        : image_size[1],
        "date_captured" : date_captured,
        "license"       : license_id,
        "coco_url"      : coco_url,
        "flickr_url"    : flickr_url
    }
    return image_info

def create_annotation_info(ann_id, img_id, cat_id, area, bounding_box, trk_id, inflight):
    annotation_info = {
        "id"            : ann_id,
        "image_id"      : img_id,
        "category_id"   : cat_id,        # object label
        "iscrowd"       : 0,
        "area"          : area,          # float
        "bbox"          : bounding_box,  # [x,y,width,height]
        "segmentation"  : [],            # [polygon]
        "attributes"    : {
            "inflight"  : inflight,
            "occluded"  : False,
            "rotation"  : 0.0,
            "track_id"  : trk_id,        # track id based on the start frame id in a track sequence
            "keyframe"  : True
        }
    }
    return annotation_info

def get_coco_metadata(dataset_type="train"):
    coco_output = {}
    coco_output["info"] = {
        "description": "Manacus COCO Keyframe Dataset - {}".format(dataset_type),
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

def get_track_interpolated_frames(track_item, frame_step_size, frame_ext_type):
    """ 
    TODO: Interpolate the bounding box with intermittent keyframes 
    """
    # Parse each keyframe annotation associated with this track
    for shape_idx, shape_item in enumerate(track_item['shapes']):
        if frame_ext_type == "FFMPEG":
            # FFMPEG extraction frame ids start from 1 => frame_id = frame_id + 1
            frame_id = shape_item['frame'] * frame_step_size + 1
        else:
            # OpenCV extraction frame ids start from 0 
            frame_id = shape_item['frame'] * frame_step_size

        if shape_item['outside'] == True:
            # Annotation keyframes outside the track sequence. Not a keyframe so skip this.
            # Keep the previous frame identifier for end frame labels
            continue

    return 

def parse_task_track_interpolated_annotations(ann_data, img_basename, width=None, height=None, frame_step_size=1, frame_ext_type="FFMPEG"):
    """
    TODO: Extract annotations by interpolating bounding box labels between keyframes in a track sequence
    """
    task_frame_ids  = set()
    gts_coco_labels = {"images":[], "annotations":[]}
    pbar = tqdm()
    for track_idx, track_item in enumerate(ann_data[0]['tracks']):
        trk_id = COUNT_ID_DICT["trk_id"] = COUNT_ID_DICT["trk_id"] + 1
        label_name  = track_item['label']
        # AttributeSet#1 labels associated with this track_id
        is_inflight = get_attributes(track_item['attributes'], key="inflight")
        track_frames = get_track_interpolated_frames(track_item, frame_step_size, frame_ext_type)
        


    return task_frame_ids, gts_coco_labels

def parse_task_track_keyframe_annotations(ann_data, img_basename, width=None, height=None, frame_step_size=1, frame_ext_type="FFMPEG"):
    """
    Extract the category label for the objects for each keyframe in a track sequence
    """
    vid_images_dict = dict()
    task_frame_ids  = set()
    task_label_ids  = []
    gts_coco_labels = {"images":[], "annotations":[]}
    pbar = tqdm()
    for track_idx, track_item in enumerate(ann_data[0]['tracks']):
        trk_id = COUNT_ID_DICT["trk_id"] = COUNT_ID_DICT["trk_id"] + 1
        label_name  = track_item['label']
        # AttributeSet#1 labels associated with this track_id
        is_inflight = get_attributes(track_item['attributes'], key="inflight")

        # Parse each keyframe annotation associated with this track
        for shape_idx, shape_item in enumerate(track_item['shapes']):
            if frame_ext_type == "FFMPEG":
                # FFMPEG extraction frame ids start from 1 => frame_id = frame_id + 1
                frame_id = shape_item['frame'] * frame_step_size + 1
            else:
                # OpenCV extraction frame ids start from 0 
                frame_id = shape_item['frame'] * frame_step_size

            if shape_item['outside'] == True:
                # Annotation keyframes outside the track sequence. Not a keyframe so skip this.
                # Keep the previous frame identifier for end frame labels
                continue
            
            # Images unique id in gts_coco_labels["images"] dict.         
            key_img_name   = "{}-{:08d}.jpg".format(img_basename, frame_id)
            if key_img_name in vid_images_dict.keys():
                image_info = vid_images_dict[key_img_name]
                image_id   = image_info["id"]
            else:
                image_id   = COUNT_ID_DICT["img_id"] = COUNT_ID_DICT["img_id"] + 1
                # COCO Format Image entry if not in dict for image metadata reuse
                image_info = create_image_info(image_id, key_img_name, image_size=[width, height])
                vid_images_dict[key_img_name] = image_info
                gts_coco_labels["images"].append(image_info)

            bb_left, bb_top       = shape_item['points'][0], shape_item['points'][1]
            bb_width, bb_height   = shape_item['points'][2]-bb_left, shape_item['points'][3]-bb_top
            area   = bb_width * bb_height
            box    = [bb_left, bb_top, bb_width, bb_height]
            cat_id = MANACUS_CLASS_LABELS[label_name]["id"]
            ann_id = COUNT_ID_DICT["ann_id"] = COUNT_ID_DICT["ann_id"] + 1
            ann_info  = create_annotation_info(ann_id, image_id, cat_id, area, box, trk_id, is_inflight)
            gts_coco_labels["annotations"].append(ann_info)
            task_label_ids.append(label_name) 
            # Unique frame identifier for frame extraction
            task_frame_ids.add(frame_id)
            pbar.update(1) 
    COUNT_ID_DICT["labels"].append(task_label_ids)
    print("Track Count: {}\t > Images: {}, Labels: {} ".format(len(ann_data[0]['tracks']), len(task_frame_ids), len(task_label_ids)))
    pbar.close() 
    return task_frame_ids, gts_coco_labels

def export_task_coco(task_dir, src_images_dir, output_dir, split_type):
    gts_coco_dict = None
    
    # Read task level properties
    task_file = os.path.join(task_dir, "task.json")
    split_sub = read_json(task_file, key="subset")
    task_vidn = read_json(task_file, key="name")
    frame_step_size = get_frame_step_size(read_json(task_file, key="data")) # CVAT step frame configuration yielded every 5th frame labeling
    # Read video file data
    vid_file = os.path.join(task_dir, "data", task_vidn)
    frame_extractor = FrameExtractor(vid_file)
    
    # Read and process annotations
    ann_file = os.path.join(task_dir, "annotations.json")
    ann_data = read_json(ann_file)
    image_basename = "{}-{}".format(get_clean_basename(task_vidn), os.path.basename(task_dir))
    print(task_vidn, "\t", split_sub, "\t", frame_step_size, "\t",  vid_file)
    keyframe_ids, gts_coco_dict = parse_task_track_keyframe_annotations(ann_data, image_basename, width=frame_extractor.width, height=frame_extractor.height, frame_step_size=frame_step_size)
    # Copy the key frame identifiers in annotations
    copy_extracted_keyframe_ids(image_basename, keyframe_ids, src_images_dir, os.path.join(output_dir, split_type))
    frame_extractor.cap.release()
    return gts_coco_dict

# Extract tasks from project backup
def cocofy_project_backup_tasks(project_dir, images_dir, output_dir, split_type):
    ann_dict = get_coco_metadata(split_type)
    for dirname in os.listdir(project_dir):
        # check whether the current object is a task folder or not
        if dirname.startswith("task") and os.path.isdir(os.path.join(project_dir, dirname)):
            task_dir = os.path.join(project_dir, dirname)
            gts_coco_dict = export_task_coco(task_dir, images_dir, output_dir, split_type)
            if gts_coco_dict is not None:
                ann_dict["images"]      = ann_dict["images"]      + gts_coco_dict["images"]
                ann_dict["annotations"] = ann_dict["annotations"] + gts_coco_dict["annotations"]
    # Write accumulated annotations to JSON file
    write_json_file(os.path.join(output_dir, 'annotations', '{}.json'.format(split_type.lower())), ann_dict)
    return ann_dict["images"], ann_dict["annotations"]

"""
Usage:
- time python dataset_cvat_cocofy_key.py
"""
if __name__ == "__main__":
    # Cvat backup project unzipped at this folder with tasks as subdirectories
    project_dir = "/home/rahul/workspace/data/fcat/fcat-manacus-sample/"
    # Expected image names in directory f"{vid_basename}-{task_basename}-{:08d}.jpg"
    images_dir  = "/home/rahul/workspace/data/fcat/fcat-manacus-sample/frames"
    # Output COCO directory
    #output_dir  = "/home/rahul/workspace/data/fcat/coco-fcat-manacus-v3/"
    output_dir  = "/home/rahul/workspace/vision/manacus-dynamics/dataset/fcat/coco/fcat-manacus-v3/"
    # Execute extraction and cocofication for all tasks in project
    images, annotations = cocofy_project_backup_tasks(project_dir, images_dir, output_dir, split_type="all")
    print("Images: {}, Annotations: {}".format(len(images), len(annotations)))
