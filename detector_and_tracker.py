import cv2
from centroidtracker import CentroidTracker
from detector import ObjectDetector
import numpy as np
import tensorflow as tf
from stream import ThreadedCam
from trackedobjects import TrackableObject
from sort import Sort
from time import time
from numba import jit
from utils import roiselector


det = ObjectDetector('cardetector')
ct = CentroidTracker()
trackers = []
totalframes = 0
skipframes = 5
trackedobjects = {}
roiwindow = 'Pls Select ROI'

H = None
W = None

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

@jit
def countinroi(image,vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,vertices,1)
    masked = cv2.bitwise_and(image,mask)
    return np.sum(masked)

cap = cv2.VideoCapture('./samples/traffic.mp4')


ret,frame = cap.read()
H, W = frame.shape[:2]
roicoords = roiselector.ROISelector(frame)()
roicoords = np.array([roicoords],dtype=np.int32)
print(roicoords)
with tf.Session(graph = det.graph) as sess:
    while 10:
        status = 'WAITING'
        ret, frame = cap.read()
        locations = np.zeros(frame.shape[:2],dtype=np.int32)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        centroids = []
        if totalframes % skipframes == 0:
            status = 'DETECTING'
            trackers = []
            lasttime = time()
            cents,boxes,_ = det.detect_centroids(frame,sess)
            spent = time() - lasttime
            print("Spent {} second(s) on detection".format(spent))
            lasttime = time()
            for box in boxes:
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame,tuple(box))
                trackers.append(tracker)
            spent = time() - lasttime
            print("Spent {} second(s) on registering trackers".format(spent))
        else:
            status = 'TRACKING'
            lasttime = time()
            centroids = [[] for _ in range(len(trackers))]
            for i,tracker in enumerate(trackers):
                success, box = tracker.update(frame)
                cent = [box[0]+box[2]/2,box[1]+box[3]/2]
                centroids[i] = cent
            spent = time() - lasttime
            print("Spent {} second(s) on updating trackers".format(spent))
        lasttime = time()
        objects = ct.update(np.array(centroids,dtype=np.int32))
        spent = time() - lasttime
        print("Spent {} second(s) on updating cent trackers".format(spent))
        cv2.polylines(frame,roicoords,isClosed=False,color=[255,0,0],thickness=2)
        lasttime = time()
        for objid,centroid in objects.items():
            x,y = centroid
            x,y = check_and_correct_boundaries(x,y,0,0,W,H)
            locations[y,x] = 1
            trackedobject = trackedobjects.get(objid,None)
            if trackedobject is None:
                trackedobject = TrackableObject(objid,centroid)
            else:
                trackedobject.centroids = centroid
            trackedobjects[objid] = trackedobject
            text = "ID {}".format(objid)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        spent = time() - lasttime
        print("Spent {} second(s) on drawing and updating tracked objects".format(spent))
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        lasttime = time()
        count = countinroi(locations,roicoords)
        spent = time() - lasttime
        print("Spent {} second(s) on roi counting".format(spent))
        count_text = "Count : {}".format(count)
        cv2.putText(frame, status, (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, count_text, (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Result",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        totalframes += 1

cap.release()
cv2.destroyAllWindows()