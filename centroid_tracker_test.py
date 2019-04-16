from detector import ObjectDetector
from centroidtracker import CentroidTracker
import cv2
import numpy as np
import tensorflow as tf
import stream

det = ObjectDetector('cardetector')
tracker = CentroidTracker()

cap = stream.ThreadedCam('samples/Road traffic video for object recognition.mp4',fps=30).start()

with tf.Session(graph = det.graph) as sess:
    while 2 :
        frame = cap.get_frame()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        centroids,boxes = det.detect_centroids(frame,sess)
        objects = tracker.update(centroids)
        print(objects)
        for objid,centroid in objects.items():
            y,x = centroid
            text = "ID {}".format(objid)
            cv2.putText(frame, text, (int(x) - 10, int(y) - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame,(int(x),int(y)),radius=5,color=[0,255,255],thickness=-1)
        cv2.imshow("Test",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.stop()
cv2.destroyAllWindows()
