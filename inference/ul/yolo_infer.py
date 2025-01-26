# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import argparse
import datetime

import cv2
import torch
import pandas as pd
import numpy as np
import onnxruntime as ort

from tqdm import tqdm

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


class YOLOv11Handler:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, confidence_thres, iou_thres):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        try:
            # Check the requirements and select the appropriate backend (CPU or GPU)
            device = ort.get_device()
            cuda = True if device == 'GPU' else False
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            so = ort.SessionOptions()
            so.log_severity_level = 3

            # Create an inference session using the ONNX model and specify execution providers
            self.session = ort.InferenceSession(self.onnx_model, providers=providers, sess_options=so)
            # Get the model inputs
            self.model_inputs = self.session.get_inputs()
            # Store the shape of the input for later use
            input_shape = self.model_inputs[0].shape
            self.input_width = input_shape[2]
            self.input_height = input_shape[3]
        except Exception as e:
            raise Exception(f"Cannot load model {onnx_model}: {e}")
        
        # Load the class names from the COCO dataset
        self.classes = {  
            0 : "Male",
            1 : "Female",
            2 : "Unknown"
        }

        # Generate a color palette for the classes
        self.color_palette = [
            [51,221,255], 
            [240,120,240], 
            [250,250,55]
        ]


    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


    def preprocess(self, input_image):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        if isinstance(input_image, str):
            # Read the input image using OpenCV
            self.img = cv2.imread(input_image)
        else:
            self.img = input_image

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data


    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        results = {
                    "label_id": [],
                    "label_name": [],
                    "confidence": [],
                    "box_xtl" : [], 
                    "box_ytl" : [], 
                    "box_xbr" : [], 
                    "box_ybr" : [],
                }
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)

            # Add to output
            results["label_id"].append(class_id)
            results["confidence"].append(score)
            results["label_name"].append(self.classes[class_id])
            x1, y1, w, h = box
            results["box_xtl"].append(x1)
            results["box_ytl"].append(y1)
            results["box_xbr"].append(x1+w)
            results["box_ybr"].append(y1+h)

        # Return the modified input image
        return input_image, results


    def infer(self, input_image):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """

        # Preprocess the image data
        img_data = self.preprocess(input_image)

        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(self.img, outputs)  # output image



"""
Input: Image
    - time python yolo_infer.py --view-debug --image "./deploy/frame_000830.PNG"
Input: Video
    - time python yolo_infer.py --view-debug --video "./deploy/LM.P4_1.8.22-1.13.22_0127.MP4"

- Intermediate output with switch '--view-debug':
    - *.v03.result.{jpg or mp4} : Video with bounding box draw inframe
    - *.v03.result.csv          : CSV file with frame recognition information written in rows

"""
if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       type=str,      default=None,  help="Path where video file is located")
    parser.add_argument("--image",       type=str,      default=None,  help="Path where image file is located")
    parser.add_argument("--model",       type=str,      default="./deploy/best_y11m-dv5-default-e10.onnx",  help="Path where ONNX model is located")

    parser.add_argument("--out-suffix",  type=str,      default="v03.result",  help="Result filename suffix")
    parser.add_argument("--out-path",    type=str,      default="./results",  help="Result output path")

    parser.add_argument("--conf-thres",  type=float,    default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres",   type=float,    default=0.5, help="NMS IoU threshold")
    parser.add_argument('--view-debug',  action='store_true', help='write qualitative intermediate results')
    args = parser.parse_args()
    # Setup Parameters 
    np.random.seed(0)

    # Create an instance of the YOLOv8 class with the specified arguments
    model = YOLOv11Handler(args.model, args.conf_thres, args.iou_thres)

    result_out_file = ""
    result_df = pd.DataFrame()
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
                #results = model.infer(img_from_frame, thresholds)
                _, results = model.infer(img_from_frame)
                # Prepare a result item
                df = pd.DataFrame(results)
                df["frame_id"] = i
                result_df_list.append(df)
                # Visualize intermediate qualitative results in video
                if args.view_debug:
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
            # Perform object detection and obtain the output image
            output_image, results = model.infer(args.image)
            # Prepare results
            result_df = pd.DataFrame(results)
            result_df["frame_id"] = args.image
            result_df.to_csv("{}.{}.csv".format(result_out_base, out_suffx), index=False)
            # Visualize results
            if args.view_debug:
                result_out_file = "{}.{}.jpg".format(result_out_base, out_suffx)
                cv2.imwrite(result_out_file, output_image)
            print("{} - {} results written to {}".format(datetime.datetime.now(), len(result_df), "{}.{}.csv".format(result_out_base, out_suffx)))
        except Exception as e:
            raise Exception(f"Unable to process {args.image}: {e}")
    else:
        print("Nothing to do")


