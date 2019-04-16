import cv2
from centroidtracker import CentroidTracker
from detector import ObjectDetector
import numpy as np
import tensorflow as tf
from stream import ThreadedCam
from trackedobjects import TrackableObject


det = ObjectDetector('cardetector')
ct = CentroidTracker()
trackers = []
totalframes = 0
skipframes = 10
trackedobjects = {}
roicoords = []
takepoints = True
roiwindow = 'Pls Select ROI'

H = None
W = None

def check_and_correct_boundaries(x,y,xmin,ymin,xmax,ymax):
    if x < xmin:
        x = xmin
    elif x > xmax:
        x = xmax
    if y < ymin:
        y = ymin
    elif y > ymax:
        y = ymax
    return x,y    

def countinroi(image,vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,vertices,1)
    masked = cv2.bitwise_and(image,mask)
    return np.sum(masked)

def get_coords_from_clicks(event,x,y,flags,param):
    global roicoords, takepoints
    if event == cv2.EVENT_LBUTTONUP:
        roicoords.append((x,y))
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        roicoords.append((x,y))
        takepoints = False

cap = cv2.VideoCapture('./samples/traffic.mp4')


ret,frame = cap.read()
H, W = frame.shape[:2]
cv2.namedWindow(roiwindow)
cv2.setMouseCallback(roiwindow,get_coords_from_clicks)
while takepoints:
    cv2.imshow(roiwindow,frame)
    key = cv2.waitKey(1) & 0xFF
cv2.destroyAllWindows()

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
            cents,boxes = det.detect_centroids(frame,sess)
            for box in boxes:
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame,tuple(box))
                trackers.append(tracker)
        else:
            status = 'TRACKING'
            for tracker in trackers:
                success, box = tracker.update(frame)
                cent = [box[0]+box[2]/2,box[1]+box[3]/2]
                centroids.append(cent)
        print(status)
        print(centroids)
        objects = ct.update(np.array(centroids,dtype=np.int32))
        cv2.polylines(frame,roicoords,isClosed=False,color=[255,0,0],thickness=2)
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
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        count = countinroi(locations,roicoords)
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