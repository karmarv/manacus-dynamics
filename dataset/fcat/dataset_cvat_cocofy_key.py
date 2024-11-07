import csv
import json
import shutil
import os, glob
import argparse
import datetime

import cv2 as cv
import numpy as np
from scipy.interpolate import interp1d
from ineye import viz

from tqdm import tqdm

# labels for Manacus 
MANACUS_CLASS_LABELS={
        "Male"      : { "id": 1 , "group": "manacus" }, 
        "Female"    : { "id": 2 , "group": "manacus" },
        "Unknown"   : { "id": 3 , "group": "manacus"  }
}

# Counts state information
COUNT_ID_DICT = { "img_id":0, "ann_id":0, "trk_id":0, "Male":0, "Female":0, "Unknown":0, "labels":[] }

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
    viz.write_list_file(list_file, frame_files)
    return 


def copy_extracted_frame_ids(basename, frame_ids, src_images_dir, output_dir):
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
            copy_link_files(src=frame_path, dst=os.path.join(dst_images_dir, framename))
            frame_files.append([os.path.join("images", framename)])
        else:
            msg = "Id: {}, Invalid Frame: {}, ".format(frame_id, frame_path)
            print(msg)
            raise Exception(msg)
    # Write frame path to file
    list_file = os.path.join(output_dir, 'images.txt')
    viz.write_list_file(list_file, frame_files)
    return 



# File helper utilities
def read_json(filename, key=None):
    data = {}
    with open(filename) as jf:
        data = json.load(jf)
    if key is not None:
        return data[key]
    else:
        return data


def copy_link_files(src, dst, symlink=True):
    """ Create or copy a symlink file"""
    if os.path.islink(src):
        linkto = os.readlink(src)
        os.symlink(linkto, dst)
    elif symlink:
        os.symlink(src, dst)
    else:
        shutil.copy2(src,dst)

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

def create_annotation_info(ann_id, img_id, cat_id, area, bounding_box, trk_id, inflight, keyframe=True):
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
            "keyframe"  : keyframe
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


def get_track_interpolated_frame_boxes(track_item, frame_step_size, frame_ext_type, drop_last_frames_count):
    """ 
    Interpolate the bounding box with coarse keyframe identifiers
    """
    sparse_bboxes = []
    key_frame_ids = []
    outside_frame_id = None
    # Parse each keyframe annotation associated with this track
    for shape_idx, shape_item in enumerate(track_item['shapes']):
        if frame_ext_type == "FFMPEG":                          # FFMPEG extraction frame ids start from 1 => frame_id = frame_id + 1
            frame_id = shape_item['frame'] * frame_step_size + 1
        else:                                                   # OpenCV extraction frame ids start from 0 
            frame_id = shape_item['frame'] * frame_step_size
        # compute bounds of the keyframe label
        bb_left, bb_top       = shape_item['points'][0], shape_item['points'][1]
        bb_width, bb_height   = shape_item['points'][2]-bb_left, shape_item['points'][3]-bb_top
        area   = bb_width * bb_height
        box    = [frame_id, bb_left, bb_top, bb_width, bb_height, area]
        sparse_bboxes.append(box)
        if (shape_item['outside'] == True) or (shape_idx >= len(track_item['shapes'])-1):
            # Annotation outside the track sequence or reached end of sequence 
            # Interpolate until here but delete this frame from sequence later
            outside_frame_id = frame_id
        else:
            key_frame_ids.append(frame_id)
    # Return if interpolation is not needed
    if len(sparse_bboxes) == 1:
        sparse_bboxes[0].extend([1])
        return sparse_bboxes
    elif len(sparse_bboxes) < 1:
        return sparse_bboxes
    
    """ Fit numpy array based on frame_id sequence """
    coarse_y_np = np.array(sparse_bboxes)
    coarse_x    = np.array(coarse_y_np[:, 0]).astype(int)
    #for item in coarse_y_np.tolist():
    #    print("Old: ", item)
    # New frame sequence that needs to be filled based on the frame_id 
    new_frames = np.linspace(int(coarse_y_np[:, 0].min()), int(coarse_y_np[:, 0].max()), int(coarse_y_np[:, 0].max()-coarse_y_np[:, 0].min()) + 1)
    
    # Apply the interpolation to each column
    fit = interp1d(coarse_x, coarse_y_np, axis=0) # , fill_value="extrapolate"
    new_sbboxes_np = fit(new_frames)
    if outside_frame_id is not None:
        new_sbboxes = []
        outside_frame_ids = [outside_frame_id-i for i in range(drop_last_frames_count)]
        for item in new_sbboxes_np.tolist():
            # Avoid the last few outside switched frame
            frame_id = int(item[0])
            if frame_id not in outside_frame_ids: 
                if frame_id in key_frame_ids:
                    item.extend([1])        # Keyframe=True
                else:
                    item.extend([0])        # Keyframe=False
                new_sbboxes.append(item)
        return new_sbboxes
    return new_sbboxes_np.tolist()


def parse_task_track_interpolated_annotations(ann_data, img_basename, width=None, height=None, frame_step_size=1, frame_ext_type="FFMPEG", drop_last_frames_count = 4):
    """
    Extract annotations by interpolating bounding box labels between keyframes in a track sequence
    """
    vid_images_dict = dict()
    task_frame_ids  = set()
    task_label_ids  = []
    gts_coco_labels = {"images":[], "annotations":[]}
    pbar = tqdm()
    for track_idx, track_item in enumerate(ann_data[0]['tracks']):
        trk_id = COUNT_ID_DICT["trk_id"] = COUNT_ID_DICT["trk_id"] + 1
        label_name  = track_item['label']
        category_id = MANACUS_CLASS_LABELS[label_name]["id"]
        # AttributeSet#1 labels associated with this track_id
        is_inflight = get_attributes(track_item['attributes'], key="inflight")
        trak_frames = get_track_interpolated_frame_boxes(track_item, frame_step_size, frame_ext_type, drop_last_frames_count)
        for frame_box in trak_frames:
            #print(frame_box[0], " ", frame_box[1:5], "\t", frame_box[-2], "\t", frame_box[-1])
            frame_id  = int(frame_box[0])
            box, area = frame_box[1:5], frame_box[-2]
            keyframe  = True if int(frame_box[-1]) == 1 else False
            
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
            
            # Create COCO annotations entry
            ann_id = COUNT_ID_DICT["ann_id"] = COUNT_ID_DICT["ann_id"] + 1
            ann_info  = create_annotation_info(ann_id, image_id, category_id, area, box, trk_id, is_inflight, keyframe)
            gts_coco_labels["annotations"].append(ann_info)
            # Unique frame identifier for frame extraction
            task_frame_ids.add(frame_id)
            task_label_ids.append(label_name) 
            COUNT_ID_DICT[label_name] = COUNT_ID_DICT[label_name] + 1
            pbar.update(1) 

    COUNT_ID_DICT["labels"].append(task_label_ids)
    print("Track Count: {}\t > Images: {}, Labels: {} ".format(len(ann_data[0]['tracks']), len(task_frame_ids), len(task_label_ids)))
    pbar.close() 
    return task_frame_ids, gts_coco_labels


@DeprecationWarning
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
        category_id = MANACUS_CLASS_LABELS[label_name]["id"]
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
            
            ann_id = COUNT_ID_DICT["ann_id"] = COUNT_ID_DICT["ann_id"] + 1
            ann_info  = create_annotation_info(ann_id, image_id, category_id, area, box, trk_id, is_inflight)
            gts_coco_labels["annotations"].append(ann_info)
            task_label_ids.append(label_name) 
            COUNT_ID_DICT[label_name] = COUNT_ID_DICT[label_name] + 1
            # Unique frame identifier for frame extraction
            task_frame_ids.add(frame_id)
            pbar.update(1) 
    COUNT_ID_DICT["labels"].append(task_label_ids)
    print("Track Count: {}\t > Images: {}, Labels: {} ".format(len(ann_data[0]['tracks']), len(task_frame_ids), len(task_label_ids)))
    pbar.close() 
    return task_frame_ids, gts_coco_labels

def export_task_coco(task_dir, src_images_dir, output_dir, split_type, keyframes_only):
    """ Read the task annotations and link the relevant images"""
    gts_coco_dict = None
    # Read task level properties
    task_file = os.path.join(task_dir, "task.json")
    split_sub = read_json(task_file, key="subset")
    task_vidn = read_json(task_file, key="name")
    frame_step_size = get_frame_step_size(read_json(task_file, key="data")) # CVAT step frame configuration yielded every 5th frame labeling
    # Read video file data
    vid_file = os.path.join(task_dir, "data", task_vidn)
    frame_extractor = viz.FrameExtractorFfmpeg(vid_file)
    
    # Read and process annotations
    ann_file = os.path.join(task_dir, "annotations.json")
    ann_data = read_json(ann_file)
    image_basename = "{}-{}".format(viz.get_clean_basename(task_vidn), os.path.basename(task_dir))
    print(task_vidn, "\t", split_sub, "\t", frame_step_size, "\t",  vid_file)
    if keyframes_only:
        frame_ids, gts_coco_dict = parse_task_track_keyframe_annotations(ann_data, image_basename, width=frame_extractor.width, height=frame_extractor.height, frame_step_size=frame_step_size)
    else:
        frame_ids, gts_coco_dict = parse_task_track_interpolated_annotations(ann_data, image_basename, width=frame_extractor.width, height=frame_extractor.height, frame_step_size=frame_step_size)
    # Copy the frame identifiers referenced in annotations
    copy_extracted_frame_ids(image_basename, frame_ids, src_images_dir, os.path.join(output_dir, split_type))
    return gts_coco_dict


# Extract tasks from project backup
def cocofy_project_backup_tasks(project_dir, images_dir, output_dir, split_type, keyframes_only):
    ann_dict = get_coco_metadata(split_type)
    for dirname in os.listdir(project_dir):
        # check whether the current object is a task folder or not
        if dirname.startswith("task") and os.path.isdir(os.path.join(project_dir, dirname)):
            task_dir = os.path.join(project_dir, dirname)
            gts_coco_dict = export_task_coco(task_dir, images_dir, output_dir, split_type, keyframes_only)
            if gts_coco_dict is not None:
                ann_dict["images"]      = ann_dict["images"]      + gts_coco_dict["images"]
                ann_dict["annotations"] = ann_dict["annotations"] + gts_coco_dict["annotations"]
    # Write accumulated annotations to JSON file
    viz.write_json_file(os.path.join(output_dir, 'annotations', '{}.json'.format(split_type.lower())), ann_dict)
    return ann_dict["images"], ann_dict["annotations"]


"""
Usage:
# test runs
- conda activate mana && cd ~/workspace/vision/manacus-dynamics/dataset/fcat

# V3 dataset (with interpolation)
- time python dataset_cvat_cocofy_key.py --project-backup "/home/rahul/workspace/data/fcat/fcat-manacus-sample/" --images-dir "/home/rahul/workspace/data/fcat/fcat-manacus-sample/frames" --output-dir "/home/rahul/workspace/vision/manacus-dynamics/dataset/fcat/coco/fcat-manacus-v3-inter/" 

# V4 dataset (with interpolation)
- time python dataset_cvat_cocofy_key.py --project-backup "/home/rahul/workspace/data/fcat/fcat-manacus-videos-cvat-project-backup/" --images-dir "/home/rahul/workspace/data/fcat/fcat-manacus-videos-cvat-project-backup/frames/" --output-dir "/home/rahul/workspace/vision/manacus-dynamics/dataset/fcat/coco/fcat-manacus-v4-inter/"

"""
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Process CVAT project backup')
    parser.add_argument("--project-backup",   type=str,  default="/home/rahul/workspace/data/fcat/fcat-manacus-sample/",  help="Cvat backup project unzipped at this folder with tasks as subdirectories")
    parser.add_argument("--images-dir",       type=str,  default="/home/rahul/workspace/data/fcat/fcat-manacus-sample/frames",  help="Source directory for all video frames/images. Expected image names in directory `vid_basename-task_basename-08d.jpg`")
    parser.add_argument("--output-dir",       type=str,  default="/home/rahul/workspace/vision/manacus-dynamics/dataset/fcat/coco/fcat-manacus-v3/",  help="Output directory for COCO dataset")
    args=parser.parse_args()
    # Parameterize arguments
    project_dir, images_dir, output_dir  =  args.project_backup, args.images_dir, args.output_dir
    # Execute extraction and cocofication for all tasks in project
    images, annotations = cocofy_project_backup_tasks(project_dir, images_dir, output_dir, split_type="all", keyframes_only=False)
    print("Images: {}, Annotations: {}".format(len(images), len(annotations)))
    # Write the defect histogram to file
    viz.write_json_file(os.path.join(output_dir, 'annotations', 'histogram.json'), COUNT_ID_DICT)
