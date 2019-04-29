import cv2

class ROISelector(object):

    def __init__(self,image,windowname="Select ROI"):
        self.image = image
        self.windowname = windowname
        self.roicoords = []
        self.taking = True

    def _mouse_callback(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONUP:
            self.roicoords.append((x,y))
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.roicoords.append((x,y))
            self.taking = False
    
    def __call__(self):
        cv2.namedWindow(self.windowname)
        cv2.setMouseCallback(self.windowname,self._mouse_callback)
        while self.taking:
            if len(self.roicoords) > 1:
                lastpoint = self.roicoords[-1]
                beforelast = self.roicoords[-2]
                cv2.line(self.image,beforelast,lastpoint,[0,0,255],1)
            cv2.imshow(self.windowname,self.image)
            _ = cv2.waitKey(1) & 0xFF
        cv2.destroyAllWindows()
        return self.roicoords[:-1]