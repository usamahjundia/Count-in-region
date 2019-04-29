import cv2
import roiselector

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

roi = roiselector.ROISelector(frame)()

print(roi)