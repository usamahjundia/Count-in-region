# Count-in-region

Using object detector + tracker to count the occurence of objects (e.g cars) inside a certain region of interest. 

The tracker seems a bit overkill as it is possible to just count the detection, but tracking is used so as to not run detection every single frame, and to prevent counting failure on noisy detection.
