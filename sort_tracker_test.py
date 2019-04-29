from detector import ObjectDetector
from sort import *
import cv2
import numpy as np
import tensorflow as tf
import stream
from numba import jit
from functools import partial
from centroidtracker import CentroidTracker


def check_and_correct_boundaries(x,y,xmin,ymin,xmax,ymax):
    if x < xmin:
        x = xmin
    elif x > xmax:
        x = xmax-1
    if y < ymin:
        y = ymin
    elif y > ymax:
        y = ymax-1
    return x,y

def from_boxes_to_centroids_ids(boxes):
    result = [[] for _ in range(len(boxes))]
    for i, box in enumerate(boxes):
        centx = int((box[0] + box[2]) / 2)
        centy = int((box[1] + box[3]) / 2)
        boxid = int(box[4])
        result[i] = [centx,centy,boxid]
    return result

det = ObjectDetector('cardetector')
cap = stream.ThreadedCam("samples/traffic.mp4",fps=30).start()
frame = cap.get_frame()
skip_frames = 5

H,W = frame.shape[:2]
check_boundaries = partial(check_and_correct_boundaries,xmin=0,ymin=0,xmax=W,ymax=H)
tracker = Sort(max_age=19,min_hits=0)
frame_ctr = 0
with tf.Session(graph=det.graph) as sess:
    while 25:
        frame = cap.get_frame()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if frame_ctr % skip_frames == 0:
            boxes = det.detect_boxes_and_scores(frame,sess,sort=True)
        else:
            boxes = []
        track_and_ids = tracker.update(boxes)
        track_and_ids = from_boxes_to_centroids_ids(track_and_ids)
        print(track_and_ids)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        for cents in track_and_ids:
            text = "ID {}".format(cents[2])
            cv2.putText(frame, text, (cents[0] - 10, cents[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame,(cents[0],cents[1]),3,[0,255,0],-1)
        cv2.imshow("aaaa",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cv2.destroyAllWindows()
cap.stop()