from scipy.spatial import distance as dist
import numpy as np
from collections import OrderedDict

class CentroidTracker(object):
    
    def __init__(self,maxDisappeared=10):
        self.maxdisappear = maxDisappeared
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.nextObjectID = 0
    
    def register(self,centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    
    def remove(self,_id):
        del self.objects[_id]
        del self.disappeared[_id]

    def update(self,inputcentroids):
        if len(inputcentroids) == 0:
            for objectid in list(self.disappeared.keys()):
                self.disappeared[objectid]+= 1
                if self.disappeared[objectid] > self.maxdisappear:
                    self.remove(objectid)
            return self.objects
        if len(self.objects) == 0:
            for centroid in inputcentroids:
                self.register(centroid)
            return self.objects
        else:
            trackids = list(self.objects.keys())
            trackcents = list(self.objects.values())

            distances = dist.cdist(np.array(trackcents),inputcentroids)

            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            usedrows = set()
            usedcols = set()
            for row, col in zip(rows,cols):
                if row in usedrows or col in usedcols:
                    continue
                trackid = trackids[row]
                self.objects[trackid] = inputcentroids[col]
                self.disappeared[trackid] = 0
                usedrows.add(row)
                usedcols.add(col)
            unusedrows = set(range(0,distances.shape[0])).difference(usedrows)
            unusedcols = set(range(0,distances.shape[1])).difference(usedcols)
            if distances.shape[0] >= distances.shape[1]:
                for row in unusedrows:
                    objid = trackids[row]
                    self.disappeared[objid] += 1
                    if self.disappeared[objid] > self.maxdisappear:
                        self.remove(objid)
            else:
                for col in unusedcols:
                    self.register(inputcentroids[col])
            return self.objects
                