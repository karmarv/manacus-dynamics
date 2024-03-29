import cv2
import os
from glob import glob

def get_info(video_path:str):
    # read a informations
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height, length


def frame_resizing(width:int, height:int, frame_size) -> list:
    if width > height:
        aspect_ratio = width / height
        if height >= frame_size:
            height = frame_size
        width = int(aspect_ratio*height)
    else:
        aspect_ratio = height / width
        if width >= frame_size:
            width = frame_size
        height = int(aspect_ratio*width)
    return [width, height]