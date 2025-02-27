import random
import csv
import os, json
import argparse
import funcy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np
import shutil

# Configuration and path
random.seed(0) 

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    #print("Sample annotations:",annotations)
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def filter_images(images, annotations):
    #print("Sample images:",images)
    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)
    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)

"""
Write list to file or append if it exists
"""
def write_list_file(filename, rows, delimiter=','):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a+") as my_csv:
        csvw = csv.writer(my_csv,delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC)
        csvw.writerows(rows)


def copy_files(src, dst, symlink=True):
    """ Create or copy a symlink file"""
    if os.path.islink(src):
        linkto = os.readlink(src)
        os.symlink(linkto, dst)
    elif symlink:
        os.symlink(src, dst)
    else:
        shutil.copy2(src,dst)

def save_images(images, dst_folder, src_folder):
    count=1
    img_names = []
    os.makedirs(dst_folder, exist_ok=True)
    for img in tqdm(images):
        src = os.path.join(src_folder, img["file_name"])
        dst = os.path.join(dst_folder, img["file_name"])
        img_names.append([img["file_name"]])
        #shutil.copyfile(src, dst)
        copy_files(src, dst)
        count+=1
    # Save the files that are copied
    write_list_file(os.path.join(dst_folder, "..", "images.txt"), img_names)
    return count

parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('train', type=str, help='Where to store COCO training annotations')
parser.add_argument('test', type=str, help='Where to store COCO test annotations')
parser.add_argument('-s', dest='split', type=float, required=True,
                    help="A percentage of a training set split; a number in (0, 1)")
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')

parser.add_argument('--multi-class', dest='multi_class', action='store_true',
                    help='Split a multi-class dataset while preserving class distributions in train and test sets')

args = parser.parse_args()



def main(args):
    ann_name, ann_extn = os.path.splitext(os.path.basename(args.annotations))
    output_dir = os.path.join(os.path.dirname(args.annotations), "..")
    images_dir = os.path.join(output_dir, ann_name, "images")
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)
        print("Reading annotations for {} images".format(number_of_images))

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)


        if args.multi_class:

            annotation_categories = funcy.lmap(lambda a: int(a['category_id']), annotations)

            #bottle neck 1
            #remove classes that has only one sample, because it can't be split into the training and testing sets
            annotation_categories =  funcy.lremove(lambda i: annotation_categories.count(i) <=1  , annotation_categories)

            annotations =  funcy.lremove(lambda i: i['category_id'] not in annotation_categories  , annotations)
            X_train, y_train, X_test, y_test = iterative_train_test_split(np.array([annotations]).T,np.array([ annotation_categories]).T, test_size = 1-args.split)

            # Save COCO annotations            
            train_images = filter_images(images, X_train.reshape(-1))
            save_coco(args.train, info, licenses, train_images, X_train.reshape(-1).tolist(), categories)

            test_images = filter_images(images, X_test.reshape(-1))
            save_coco(args.test, info, licenses, test_images, X_test.reshape(-1).tolist(), categories)
            print("Multiclass Saved {} labels in {} and {} in {}".format(len(X_train), args.train, len(X_test), args.test))

        else:

            X_train, X_test = train_test_split(np.array(images), train_size=args.split)
            anns_train = filter_annotations(annotations, X_train)
            anns_test=filter_annotations(annotations, X_test)

            # Save COCO annotations            
            train_images = X_train.reshape(-1)
            save_coco(args.train, info, licenses, X_train.reshape(-1).tolist(), anns_train, categories)

            test_images = X_test.reshape(-1)
            save_coco(args.test, info, licenses, X_test.reshape(-1).tolist(), anns_test, categories)
            print("Saved {} labels in {} and {} in {}".format(len(X_train), args.train, len(X_test), args.test))

        
        # Copy images as per split to the annotations file base data directory
        train_name, ann_extn = os.path.splitext(os.path.basename(args.train))
        os.makedirs(os.path.join(output_dir, train_name), exist_ok=True)
        train_count = save_images(train_images, os.path.join(output_dir, train_name, "images"), images_dir)
        test_name, ann_extn = os.path.splitext(os.path.basename(args.test))
        os.makedirs(os.path.join(output_dir, test_name), exist_ok=True)
        test_count = save_images(test_images, os.path.join(output_dir, test_name, "images"), images_dir)
        print("Copied {} images in {} and {} in {}".format(train_count, args.train, test_count, args.test))
            
"""
Usage: 
V3: Sample data split train:val:test ~ 80:10:10
- python dataset_coco_split.py -s 0.8 ./coco/fcat-manacus-v3/annotations/all.json ./coco/fcat-manacus-v3/annotations/train.json ./coco/fcat-manacus-v3/annotations/other.json
- python dataset_coco_split.py -s 0.5 ./coco/fcat-manacus-v3/annotations/other.json ./coco/fcat-manacus-v3/annotations/val.json ./coco/fcat-manacus-v3/annotations/test.json

V4 dataset (with interpolation): Data split train:val:test ~ 80:10:10
- time python dataset_coco_split.py -s 0.8 ./coco/fcat-manacus-v4-inter/annotations/all.json ./coco/fcat-manacus-v4-inter/annotations/train.json ./coco/fcat-manacus-v4-inter/annotations/other.json
    
    Reading annotations for 194926 images
    Saved 155940 labels in ./coco/fcat-manacus-v4-inter/annotations/train.json and 38986 in ./coco/fcat-manacus-v4-inter/annotations/other.json
    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 155940/155940 [00:09<00:00, 15897.56it/s]
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 38986/38986 [00:02<00:00, 16229.83it/s]
    Copied 155941 images in ./coco/fcat-manacus-v4-inter/annotations/train.json and 38987 in ./coco/fcat-manacus-v4-inter/annotations/other.json

    real    5m36.038s

- time python dataset_coco_split.py -s 0.5 ./coco/fcat-manacus-v4-inter/annotations/other.json ./coco/fcat-manacus-v4-inter/annotations/val.json ./coco/fcat-manacus-v4-inter/annotations/test.json

    Reading annotations for 38986 images
    Saved 19493 labels in ./coco/fcat-manacus-v4-inter/annotations/val.json and 19493 in ./coco/fcat-manacus-v4-inter/annotations/test.json
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 19493/19493 [00:00<00:00, 26378.71it/s]
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 19493/19493 [00:01<00:00, 16064.84it/s]
    Copied 19494 images in ./coco/fcat-manacus-v4-inter/annotations/val.json and 19494 in ./coco/fcat-manacus-v4-inter/annotations/test.json

    real    0m13.903s

"""

if __name__ == "__main__":
    main(args)