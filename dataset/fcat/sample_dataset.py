import os
import time

import cv2 as cv
import numpy as np
from ineye import viz


"""
Visualize the frame with control/progress bar
"""
def plot_video_frames(video_infile):
    start = time.time()
    try:
        # Initialize the Video Frame Extractor for sampling frames
        frmx = viz.FrameExtractor(video_infile)
        frame_count, fps = frmx.frame_count, frmx.fps
        print("FPS:{:.2f}, (Frames: {}, Duration {:.2f}), \t Video:{} ".format(fps, frame_count, frame_count/fps, video_infile))
        
        # OpenCV windowing functions
        cv_window_name = "FPS:{:.2f}, Frames:{}, Video:{}".format(fps, frame_count, os.path.basename(video_infile))
        def onCurrentFrameTrackbarChange(trackbarValue):
            print("Current Frames Value: {}".format(trackbarValue))
            pass
        cv.namedWindow(cv_window_name) 
        cv.createTrackbar('current-frame', cv_window_name, 1, frame_count, onCurrentFrameTrackbarChange)

        cv.namedWindow("flow") 

        # Initialization of index and frame iteration
        frame_id = 0
        bad_frames = 0
        prev_frame = frmx.image_from_frame(frame_id)
        prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        # create canvas to paint on
        hsv_canvas = np.zeros_like(prev_frame)
        # set saturation value (position 2 in HSV space) to 255
        hsv_canvas[..., 1] = 255
        while frame_id < frame_count:
            # Get the frame given its index
            curr_frame = frmx.image_from_frame(frame_id)
            curr_gray  = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)
            if curr_frame is None: # Bad frame, skip
                bad_frames += 1
                continue
            
            # Set the trackbar and show frame in opencv window
            cv.setTrackbarPos('current-frame', cv_window_name, frame_id)
            
            # Run optical flow and overlay image in window
            # compare prev frame with current frame
            flow = cv.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
            # get x and y coordinates
            magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
            # set hue of HSV canvas (position 1)
            hsv_canvas[..., 0] = angle*(180/(np.pi/2))
            # set pixel intensity value (position 3
            hsv_canvas[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

            flow_rgb = cv.cvtColor(hsv_canvas, cv.COLOR_HSV2BGR)
            cv.imshow("flow", flow_rgb)

            cv.imshow(cv_window_name, curr_frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit")
                break
            
            # Reset trackbar position to the current frame and proceed to next
            frame_id = cv.getTrackbarPos('current-frame', cv_window_name)
            frame_id = frame_id + 1
            prev_gray = curr_gray
        else:
            print("All frames exhausted")
    except KeyboardInterrupt:
        print('Interrupted ctrl-c')
    finally:
        # The following frees up resources and closes all windows
        if frmx.cap:
            frmx.cap.release()
        cv.destroyAllWindows()
        print("Completed in {} Sec \t - {}, has {} bad frames ".format(time.time()-start, video_infile, bad_frames))

    return

"""
Sample frames from videos 
- detect motion for foreground objects
- background rejection 
"""
if __name__ == "__main__":
    video_path = "./curated/femvisitation-videos/1.7.22-1.15.22_0047.MP4"
    plot_video_frames(video_path)