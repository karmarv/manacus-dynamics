import os, json
import argparse
import datetime

import cv2
import pandas as pd
import numpy as np
import torch
import onnxruntime as ort
from tqdm import tqdm

# Initialize labels for recognition
CLASS_LABELS = {  
            0 : "Male",
            1 : "Female",
            2 : "Unknown"
        }
LABEL_COLORS = [
            [51,221,255], 
            [240,120,240], 
            [250,250,55]
        ]

def get_class_thresholds(gt=0.25):
    # increment relative thresholds above global value
    class_thresholds = {  
            "Male": 0.0 + gt,
            "Female": 0.0 + gt,
            "Unknown" : 0.0 + gt
        }
    return class_thresholds


class FrameExtractor:
    """
    OpenCV utility to sample frames in a video 
    """
    def __init__(self, videopath):
        self.videopath   = videopath    
        self.cap         = cv2.VideoCapture(videopath)
        self.fps         = self.cap.get(cv2.CAP_PROP_FPS)
        self.width       = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height      = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("FPS:{:.2f}, (Frames: {}), \t Video:{} ".format(self.fps, self.frame_count, videopath))

    # Extract frame given the identifier
    def image_from_frame(self, frame_id):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        _, img = self.cap.read()
        return img
    
    

class ModelHandler:
    def __init__(self, labels, path="best.onnx"):
        self.model = None
        self.load_network(model=path)    
        self.labels = labels

    def load_network(self, model):
        device = ort.get_device()
        cuda = True if device == 'GPU' else False
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = ort.InferenceSession(model, providers=providers, sess_options=so)
            self.output_details = [i.name for i in self.model.get_outputs()]
            self.input_details = [i.name for i in self.model.get_inputs()]

            self.is_inititated = True
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def _infer(self, inputs: np.ndarray):
        try:
            img = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
            image = img.copy()
            image, ratio, dwdh = self.letterbox(image, auto=False)
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)

            im = image.astype(np.float32)
            im /= 255
            inp = {self.input_details[0]: im}
            # ONNX inference
            output = list()
            dets = self.model.run(self.output_details, inp)
            dets = np.array(dets[0])
            print(dets[0])

            # for det in detections - single batch .data[:, :4]
            boxes  = dets[:, :4]
            scores = dets[:, 4]
            labels = dets[:, -1]
            print("B: ", boxes.shape, "L: ", labels.shape, "S: ", scores.shape)
            print("B: ", boxes, "\nL: ", labels, "\nS: ", scores)
            boxes -= np.array(dwdh * 2)
            boxes /= ratio
            boxes = boxes.round().astype(np.int32)
            output.append(boxes)
            output.append(labels)
            output.append(scores)
            return output

        except Exception as e:
            print(e)

    def infer(self, image, thresholds):
        image = np.array(image)
        image = image[:, :, ::-1].copy()
        h, w, _ = image.shape
        detections = self._infer(image)

        results = []
        if detections:
            boxes = detections[0]
            labels = detections[1]
            scores = detections[2]

            for label_id, score, box in zip(labels, scores, boxes):
                label_name = self.labels.get(label_id, "Unknown")
                threshold = thresholds[label_name]
                if score >= threshold:
                    xtl = max(int(box[0]), 0)
                    ytl = max(int(box[1]), 0)
                    xbr = min(int(box[2]), w)
                    ybr = min(int(box[3]), h)

                    results.append({
                        "label": label_name,
                        "label_id": label_id,
                        "box_xtl": xtl, 
                        "box_ytl": ytl, 
                        "box_xbr": xbr, 
                        "box_ybr": ybr,
                        "confidence": score,
                    })
        return results
    
    def plot_one_box(self, bbox, img, color=None, label=None, line_thickness=3):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [np.random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1]), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)




"""
Input: Image
    - time python yolo_infer.py --view-debug --image "./deploy/frame_000830.PNG"
Input: Video
    - time python yolo_infer.py --view-debug --video "./deploy/LM.P4_1.8.22-1.13.22_0127.MP4"

- Intermediate output with switch '--view-debug':
    - *.v02.result.{jpg or mp4} : Video with bounding box draw inframe
    - *.v02.result.csv          : CSV file with frame recognition information written in rows

"""
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Process provided video for recognition.')
    parser.add_argument("--video",       type=str,      default=None,  help="Path where video file is located")
    parser.add_argument("--image",       type=str,      default=None,  help="Path where image file is located")
    parser.add_argument("--model",       type=str,      default="./deploy/best.onnx",  help="Path where ONNX model is located")
    parser.add_argument("--out-suffix",  type=str,      default="v02.result",  help="Result filename suffix")
    parser.add_argument("--out-path",    type=str,      default="./results",  help="Result output path")
    parser.add_argument("--threshold",   type=float,    default=0.5,  help="Minimum global threshold for detection results")
    parser.add_argument('--view-debug',  action='store_true', help='write qualitative intermediate results')
    args=parser.parse_args()
    
    # Inference and other analysis model initialization
    model = ModelHandler(CLASS_LABELS, path=args.model)
    thresholds = get_class_thresholds(args.threshold)

    
    result_out_file = ""
    result_df = pd.DataFrame()
    txt_count = 0
    out_suffx = args.out_suffix

    if args.video is not None:
        result_video_fps = 20
        print("{} - Process Video: {}".format(datetime.datetime.now(), args.video))
        # Setup output path
        if args.out_path is not None:
            os.makedirs(args.out_path, exist_ok=True)
            result_out_base = os.path.join(args.out_path, os.path.basename(args.video))
        else:
            result_out_base = args.video
        try:
            # Read input video frames
            frame_extractor = FrameExtractor(args.video)
            result_video_fps = frame_extractor.fps
            bad_frames = 0
            result_df_list  = []
            if args.view_debug:
                result_out_file = "{}.{}.mp4".format(result_out_base, out_suffx)
                vid_writer = cv2.VideoWriter(result_out_file, 
                                        cv2.VideoWriter_fourcc(*'mp4v'), result_video_fps, 
                                        (frame_extractor.width, frame_extractor.height))
            # Execute Inference on provided input
            for i in tqdm(range(frame_extractor.frame_count)):
                img_from_frame = frame_extractor.image_from_frame(i)
                if img_from_frame is None: # Bad frame, skip
                    bad_frames += 1
                    continue
                # Run inference
                results = model.infer(img_from_frame, thresholds)
                # Prepare a result item
                df = pd.DataFrame(results)
                df["frame_id"] = i
                result_df_list.append(df)
                # Visualize intermediate qualitative results in video
                if args.view_debug:
                    for det in results:
                        xyxy  = [det["box_xtl"], det["box_ytl"], det["box_xbr"], det["box_ybr"] ]
                        label = f'{det["label"]} {float(det["confidence"]):.2f}'
                        if label:
                            model.plot_one_box(xyxy, img_from_frame, label=label, color=LABEL_COLORS[int(det["label_id"])], line_thickness=2)
                    # Video result entry for debug view
                    vid_writer.write(img_from_frame)
            # Prepare results
            result_df = pd.concat(result_df_list)
            result_df["label_id"] = result_df["label_id"].astype(int)
            result_df.to_csv("{}.{}.csv".format(result_out_base, out_suffx), index=False)
            print("{} - {} results written to {}".format(datetime.datetime.now(), len(result_df), result_out_base))
            if args.view_debug:
                vid_writer.release()
            frame_extractor.cap.release()
        except Exception as e:
            raise Exception(f"Unable to process {args.video}: {e}")
    elif args.image is not None:
        print("{} - Process Image: {}".format(datetime.datetime.now(), args.image))
        # Setup output path
        if args.out_path is not None:
            os.makedirs(args.out_path, exist_ok=True)
            result_out_base = os.path.join(args.out_path, os.path.basename(args.image))
        else:
            result_out_base = args.image
        try:
            # Run inference
            image = cv2.imread(args.image, cv2.IMREAD_COLOR)    
            results = model.infer(image, thresholds)
            print("Results: ",results)
            # Prepare results
            result_df = pd.DataFrame(results)
            result_df["frame_id"] = args.image
            result_df.to_csv("{}.{}.csv".format(result_out_base, out_suffx), index=False)
            # Visualize results
            if args.view_debug:
                for det in results:
                    xyxy  = [det["box_xtl"], det["box_ytl"], det["box_xbr"], det["box_ybr"] ]
                    label = f'{det["label"]} {float(det["confidence"]):.2f}'
                    if label:
                        model.plot_one_box(xyxy, image, label=label, color=LABEL_COLORS[int(det["label_id"])], line_thickness=2)
                result_out_file = "{}.{}.jpg".format(result_out_base, out_suffx)
                cv2.imwrite(result_out_file, image)
            print("{} - {} results written to {}".format(datetime.datetime.now(), len(result_df), "{}.{}.csv".format(result_out_base, out_suffx)))
        except Exception as e:
            raise Exception(f"Unable to process {args.image}: {e}")
    else:
        print("Nothing to do")