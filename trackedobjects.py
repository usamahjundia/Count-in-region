import numpy as np

class TrackableObject:
    def __init__(self,objid,objcent):
        self.id = objid
        self.centroids = objcent
        self.counted = False