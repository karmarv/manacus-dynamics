import time
import shutil
import zipfile
import os, glob
import csv, json
import argparse
import traceback

import ffmpeg
import cv2 as cv
import subprocess

from tqdm import tqdm

def pprint(log_text, log_type="INFO", log_name="FE"):
    """ Print the log with timestamp """
    print("[{}] [{}][{}] - {}".format(time.strftime("%Y-%m-%dT%H:%M:%S"), log_name, log_type, log_text))

#
# FFMPEG Utilities
#
class FrameExtractorFfmpeg:
    def __init__(self, videopath):
        """ FFMPEG utility to sample frames in a video  """
        self.videopath   = videopath
        self._json  = ffmpeg.probe(self.videopath)
        self.props  = self.get_props()
        self.fps    = self.props["fps"]
        self.width  = self.props["width"]
        self.height = self.props["height"]
        if "duration" in self.props.keys():
            self.frame_count = self.props["duration"] * self.props["fps"]
            pprint("FPS:{:.2f}, (Frames: {}, Duration {:.2f} s), \t Video:{} ".format(self.fps, self.frame_count, self.props["duration"], videopath))
        else:
            pprint("FPS:{:.2f}, (Video:{} ".format(self.fps, videopath))

    def get_props(self):
        """ Video ffprobe properties """
        video_props = { "lib": "ffmpeg" }
        if 'format' in self._json:
            if 'duration' in self._json['format']:
                video_props["duration"] = float(self._json['format']['duration'])   
        if 'streams' in self._json:
            # commonly stream 0 is the video
            for s in self._json['streams']:
                if 'duration' in s:
                    video_props["duration"] = float(s['duration'])
                if 'avg_frame_rate' in s and s['codec_type'] == 'video':
                    frame_rate = s['avg_frame_rate'].split('/')
                    video_props["fps"]    = float(frame_rate[0])/float(frame_rate[1])
                    video_props["width"]  = int(s['width'])
                    video_props["height"] = int(s['height'])
        if "duration" in video_props.keys():
            video_props["frame_count"] = int(video_props["duration"] * video_props["fps"])

        #pprint("FF Probe (raw): {}".format(self._json))
        return video_props

    def export_frames(self, output_dir, suffix=None):
        """ Read each video frame and write to an image in output directory """
        # for each frame in video output below image file
        if suffix is not None:
            framename = "{}-{}-%08d.jpg".format(get_clean_basename(self.videopath), suffix)
        else:    
            framename = "{}-%08d.jpg".format(get_clean_basename(self.videopath))
        # Command
        cmd="cd \"{}\" && ffmpeg -nostats -hide_banner -i \"{}\" {}".format(output_dir, self.videopath, framename)
        pprint("{}".format(cmd))
        process = subprocess.Popen(cmd, shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
        stdout, stderr = process.communicate()
        if stderr:
            pprint(stderr)
            return stderr
        return stdout

#
# OpenCV Utilities
#
class FrameExtractorOpencv:
    def __init__(self, videopath):
        """ OpenCV utility to sample frames in a video  """
        self.videopath   = videopath    
        self.cap         = cv.VideoCapture(videopath)
        self.fps         = self.cap.get(cv.CAP_PROP_FPS)
        self.width       = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height      = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.props       = self.get_props()
        pprint("FPS:{:.2f}, (Frames: {}, Duration {:.2f} s), \t Video:{} ".format(self.fps, self.frame_count, self.frame_count/self.fps, videopath))

    def image_from_frame(self, frame_id):
        """ Extract frame given the identifier """
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
        _, img = self.cap.read()
        return img
    
    def image_frames_save(self, frame_ids, basename, output_dir):
        """ Extract list of frame given their identifiers """
        frame_files = []
        pbar = tqdm(total=self.frame_count)
        while self.cap.isOpened():
            pbar.update(1)
            frame_id = int(round(self.cap.get(cv.CAP_PROP_POS_FRAMES)))
            ret, img = self.cap.read()
            if ret and (frame_id in frame_ids):
                framename = "{}-{:08d}.jpg".format(basename, frame_id)
                frame_path = os.path.join(output_dir, framename)
                cv.imwrite(frame_path, img)  
                frame_files.append([os.path.join("images", framename)])
            elif not ret or frame_id >= self.frame_count:
                #print("Unreadable frame id:", frame_id)
                break
        pbar.close()
        return frame_files
    
    def count_frames_manual(self):
        """ Compute iteratively FPS given the full video file path """
        total, start = 0, time.time()               # loop over the frames of the video
        while True:
            (grabbed, frame) = self.cap.read()         # grab the current frame
            if not grabbed:                         # check to see if we have reached the end of the video
                break
            total += 1                              # increment the total number of frames read
        read_time = (time.time() - start)
        return total    

    def get_props(self):
        """ Video's duration in seconds, return a float number """
        video_props = { "lib": "opencv" }
        # Initialize a FrameExtractor to read video
        #video_props["duration"] = float(frame_extractor.frame_count / frame_extractor.fps)
        video_props["fps"]          = float(self.fps)
        video_props["width"]        = int(self.width)
        video_props["height"]       = int(self.height)
        video_props["frame_count"]  = int(self.frame_count)
        return video_props
    
    def export_frames(self, output_dir, suffix=None):
        """ Read each video frame and write to an image in output directory """
        frame_files = []
        pbar = tqdm(total=self.frame_count)
        while self.cap.isOpened():
            pbar.update(1)
            frame_id = int(round(self.cap.get(cv.CAP_PROP_POS_FRAMES)))
            ret, img = self.cap.read()
            if ret:
                # for each frame in video output below image file
                if suffix is not None:
                    framename = "{}-{}-{:08d}.jpg".format(get_clean_basename(self.videopath), suffix, frame_id)
                else:    
                    framename = "{}-{:08d}.jpg".format(get_clean_basename(self.videopath), frame_id)
                frame_path = os.path.join(output_dir, framename)
                cv.imwrite(frame_path, img)  
                frame_files.append(framename)
            elif not ret or frame_id >= self.frame_count:
                #pprint("End: Unreadable frame id:".format(frame_id))
                break
        pbar.close()
        self.frame_count  = len(frame_files)
        return frame_files

def get_frame_filename_by_video(vid_file, frame_id):
    return "{}-{:08d}.jpg".format(get_clean_basename(vid_file), frame_id)

def get_video_files(vid_folder, extensions=('.mp4', '.MP4', '.wmv', '.WMV', '.avi', '.AVI', '.mpg', '.MPG')):
    """ Look for video files in a folder """
    vid_files = []
    for root, dirnames, filenames in os.walk(vid_folder):
        for filename in filenames:
            if filename.endswith(extensions):
                vid_files.append(os.path.join(root, filename))
    return vid_files

#
# File IO Utility
#
def get_clean_basename(filename):
    """ Clean alphanumeric base filename string except '._-' """
    if filename is None:
        return None
    basename = os.path.basename(filename)
    clean_basename = "".join( x for x in basename if (x.isalnum() or x in "._-"))
    return clean_basename

def write_list_file(filename, rows, delimiter=','):
    """ Write list to file or append if it exists """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a+") as my_csv:
        csvw = csv.writer(my_csv,delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC)
        csvw.writerows(rows)

def write_json_file(filename, data):
    """ Write dictionary data to JSON """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



#
# CVAT Utility
#
def extract_video_frames(video_loc, out_frames_path, suffix=None):
    """Extract all frames from video using FFMPEG (or OpenCV) to out_frames_path directory"""
    video_props = {}
    if os.path.isfile(video_loc):
        # Step 1: FFMPEG 
        frame_ext_ff = FrameExtractorFfmpeg(video_loc)
        video_props  = frame_ext_ff.get_props()
        # In case of header issues in videos. Causes an error in FFmpeg export fallback on CV2 extraction
        if video_props["fps"] > 0 and video_props["fps"] < 90 and "duration" in video_props.keys(): 
            out_log = frame_ext_ff.export_frames(out_frames_path, suffix)
        else:
            frame_ext_cv = FrameExtractorOpencv(video_loc)
            out_log = frame_ext_cv.export_frames(out_frames_path, suffix)
            video_props  = frame_ext_cv.get_props()
            frame_ext_cv.cap.release()
        # Read the files in frames directory for image count
        frame_files = [name for name in os.listdir(out_frames_path) if os.path.isfile(os.path.join(out_frames_path, name)) and name.startswith(get_clean_basename(video_loc))]
        video_props["image_count"] = len(frame_files)
        pprint("Video extraction properties: {}".format(video_props))
    return video_loc, video_props

def process_task_backup_archive(task_backup, output_dir):
    vid_info = []
    video_type_list = ('.mp4', '.MP4', '.wmv', '.WMV', '.avi', '.AVI', '.mpg', '.MPG')
    try:
        # Extract archive to a local folder
        ann_file = os.path.join(task_backup, 'annotations.json')
        if os.path.isfile(ann_file):
            # Lookup video file, annotations
            actual_vid_file = None
            for root, dirnames, filenames in os.walk(os.path.join(task_backup, 'data')):
                for filename in filenames:
                    if filename.endswith(video_type_list):
                        actual_vid_file = os.path.join(root, filename)
                        break
            vid_info = [actual_vid_file, None, None, None]
            if actual_vid_file is None:
                raise Exception("No video file found in archive")
            
            # Extract frames to path
            out_frames_path = os.path.join(task_backup, "frames")
            if output_dir is not None:
                out_frames_path = os.path.join(output_dir, "frames")
            os.makedirs(out_frames_path, exist_ok=True)
            video_loc, video_props = extract_video_frames(actual_vid_file, out_frames_path, suffix=os.path.basename(task_backup))
            vid_info = [video_loc, video_props, out_frames_path, actual_vid_file]
    except Exception as e:
        pprint("ERROR - {}, {}".format(task_backup, e))
        print(traceback.format_exc())
    return vid_info


def test_extract_frame():
    frame_ids = [100, 200]
    vid_file = ""
    frame_extractor = FrameExtractorOpencv(vid_file)
    #img = frame_extractor.image_from_frame(frame_id)
    output_dir = os.path.join("./sample/coco/")
    basename = os.path.basename(vid_file)
    frame_extractor.image_frames_save(frame_ids, basename, output_dir)
    frame_extractor.cap.release()



"""
Usage:
# test runs
- conda activate mana && cd ~/workspace/vision/manacus-dynamics/dataset
- time python video_frames_utils.py --project-backup "/home/rahul/workspace/data/fcat/fcat-manacus-sample/" --output-dir "/home/rahul/workspace/data/fcat/fcat-manacus-sample/"
- time python video_frames_utils.py --project-backup "/home/rahul/workspace/data/fcat/fcat-manacus-videos-cvat-project-backup" --output-dir "/home/rahul/workspace/data/fcat/fcat-manacus-videos-cvat-project-backup"  > video_frame_extraction.log
    real    103m23.553s
    user    192m20.273s
    sys     6m53.918s
"""
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Process CVAT project backup')
    parser.add_argument("--project-backup",   type=str,  default="/home/rahul/workspace/data/fcat/fcat-manacus-sample/",  help="CVAT project backup folder path")
    parser.add_argument("--output-dir",       type=str,  default="/home/rahul/workspace/data/fcat/fcat-manacus-sample/",  help="Output directory for images. In case of None dump into the task/frames folder")
    args=parser.parse_args()

    index=1
    project_backup = args.project_backup
    task_list = [["index", "video_name", "video_properties", "video_frames_output", "task_archive_path", "source_video_path"]]
    
    # Target folder to copy all the tasks
    target_folder = args.output_dir
    os.makedirs(target_folder, exist_ok=True)
    if os.path.isdir(project_backup):
        for task_name in sorted(os.listdir(project_backup)):
            task_path = os.path.join(project_backup, task_name)
            if os.path.isdir(task_path):
                pprint("{:04d}\t - Processing: {}".format(index, task_path))
                # Create a histogram of task
                vid_info = process_task_backup_archive(task_path, target_folder)
                # Write CSV with split information
                if len(vid_info)>0:
                    task_list.append([index, 
                                    get_clean_basename(vid_info[0]), 
                                    vid_info[1],
                                    vid_info[2],
                                    task_path,     # actual task full filepath
                                    vid_info[3],   # actual video full filepath
                                    ])
                index+=1
    pprint("Read tasks: {}".format(len(task_list)-1))
    target_task_metadata = os.path.join(target_folder, "project_video_frame_extraction_metadata_{}.csv".format(get_clean_basename(project_backup)))
    write_list_file(target_task_metadata, task_list)