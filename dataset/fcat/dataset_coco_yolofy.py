import json
import shutil
import os, csv
import numpy as np
import contextlib
from pathlib import Path

from tqdm import tqdm
from collections import defaultdict


# labels for Manacus 
MANACUS_CLASS_LABELS={
        "Male"      : { "id": 1 , "group": "manacus" }, 
        "Female"    : { "id": 2 , "group": "manacus" },
        "Unknown"   : { "id": 3 , "group": "manacus"  }
}


"""
    # YoloV7/V8 format
    # gts_header = ['class', 'bb_x_center', 'bb_y_center', 'bb_width', 'bb_height', 'visibility']
    row  = [shape_item['frame'], label_id, bb_xcen/width, bb_ycen/height, bb_width/width, bb_height/height]
"""
def convert_yolov7(json_file, data_type, use_segments=False, filter_class_labels=None):
    path_labels = Path().resolve() / "yolo" / "labels" / data_type      # target folder
    path_labels.mkdir(parents=True, exist_ok=True)
    
    with open(json_file, 'rt', encoding='UTF-8') as jsonf:
        data = json.load(jsonf)
    
    # Create image dict
    images = {"%g" % x["id"]: x for x in data["images"]}
    # Create image-annotations dict
    imgToAnns = defaultdict(list)
    for ann in data["annotations"]:
        imgToAnns[ann["image_id"]].append(ann)
    
    # Write labels file
    for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
        img = images["%g" % img_id]
        h, w, image_file = img["height"], img["width"], img["file_name"]

        bboxes = []
        segments = []
        for ann in anns:
            if ann["iscrowd"]:
                continue
            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(ann["bbox"], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue
            
            if ann["category_id"] not in filter_class_labels:      # Skip category_id not in filter list
                continue

            cls = ann["category_id"] - 1  # class
            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)
            # Segments
            if use_segments:
                if len(ann["segmentation"]) > 1:
                    s = merge_multi_segment(ann["segmentation"])
                    s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                else:
                    s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                    s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                s = [cls] + s
                if s not in segments:
                    segments.append(s)

        # Write labels to txt file
        with open((path_labels / image_file).with_suffix(".txt"), "a") as file:
            for i in range(len(bboxes)):
                line = (*(segments[i] if use_segments else bboxes[i]),)  # cls, box or segments
                file.write(("%g " * len(line)).rstrip() % line + "\n")    
    
    return

"""
Write list to file or append if it exists
"""
def write_list_file(filename, rows, delimiter=','):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a+") as my_csv:
        csvw = csv.writer(my_csv, delimiter=delimiter)
        csvw.writerows(rows)

def copy_images(src_folder, data_type, symlink=True):
    fn = Path().resolve() / "yolo" / "images" / data_type      # target folder
    fn.mkdir(parents=True, exist_ok=True)
    images_list = []
    src_folder = Path(src_folder)
    for item in tqdm(src_folder.iterdir()):
        out = fn / os.path.basename(item)
        if symlink:
            #if item.is_symlink():
            Path(out.resolve()).symlink_to(item.resolve())
        else:
            shutil.copy(item, out)
        images_list.append([str(out)])
    write_list_file(Path().resolve() / "yolo" / "images" / "{}.txt".format(data_type), images_list)
    return

def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

def merge_multi_segment(segments):
    """
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s

"""
Usage: modify the data_base_dir variable to point to the base COCO dataset
python dataset_coco_yolofy.py 

# Sample Output V5 export with 3 labels
    Annotations ./coco/fcat-manacus-v5-fcat-ebird/annotations/train.json: 100%|█████████████████████████| 156363/156363 [00:17<00:00, 9186.95it/s]
    156363it [00:54, 2894.28it/s]
    Annotations ./coco/fcat-manacus-v5-fcat-ebird/annotations/val.json: 100%|███████████████████████████| 19546/19546 [00:02<00:00, 9275.30it/s]
    19546it [00:07, 2498.88it/s]
    Annotations ./coco/fcat-manacus-v5-fcat-ebird/annotations/test.json: 100%|██████████████████████████| 19547/19547 [00:01<00:00, 10278.11it/s]
    19547it [00:07, 2507.83it/s]

# Sample V6 export with 2 labels
    Annotations ./coco/fcat-manacus-v5-fcat-ebird/annotations/train.json: 100%|█████████████████████████| 156363/156363 [00:14<00:00, 10568.90it/s]
    156363it [00:54, 2880.80it/s]
    Annotations ./coco/fcat-manacus-v5-fcat-ebird/annotations/val.json: 100%|███████████████████████████| 19546/19546 [00:01<00:00, 11295.50it/s]
    19546it [00:07, 2667.98it/s]
    Annotations ./coco/fcat-manacus-v5-fcat-ebird/annotations/test.json: 100%|██████████████████████████| 19547/19547 [00:01<00:00, 11442.12it/s]
    19547it [00:07, 2671.44it/s]

"""
if __name__ == "__main__":
    #data_base_dir="./coco/fcat-manacus-v4-inter"
    data_base_dir="./coco/fcat-manacus-v5-fcat-ebird"
    class_ids = [MANACUS_CLASS_LABELS["Male"]["id"], MANACUS_CLASS_LABELS["Female"]["id"]]
    # Train
    convert_yolov7(json_file=data_base_dir+"/annotations/train.json", data_type="train", use_segments=False, filter_class_labels=class_ids)
    copy_images(src_folder=data_base_dir+"/train/images", data_type="train", symlink=True)

    # Val
    convert_yolov7(json_file=data_base_dir+"/annotations/val.json", data_type="val", use_segments=False, filter_class_labels=class_ids)
    copy_images(src_folder=data_base_dir+"/val/images", data_type="val", symlink=True)

    # Test
    convert_yolov7(json_file=data_base_dir+"/annotations/test.json", data_type="test", use_segments=False, filter_class_labels=class_ids)
    copy_images(src_folder=data_base_dir+"/test/images", data_type="test", symlink=True)
