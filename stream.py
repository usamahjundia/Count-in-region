import cv2
import threading
import time

class ThreadedCam(object):

    def __init__(self,source,fps=24):
        self.fps = fps
        self.cap = cv2.VideoCapture(source)
        self.ok, self.frame = self.cap.read()
        self.stopped = False
    
    def start(self):
        threading.Thread(target=self._get_frame).start()
        return self
    
    def stop(self):
        self.stopped = True
    
    def get_frame(self):
        return self.frame
    
    def _get_frame(self):
        while 2:
            if self.stopped:
                self.cap.release()
                break
            self.ok, self.frame = self.cap.read()
            if self.fps > 0:
                time.sleep(1/self.fps)

        