# Count-in-region

Using object detector + tracker to count the occurence of objects (e.g cars) inside a certain region of interest. 

The tracker seems a bit overkill as it is possible to just count the detection, but tracking is used so as to not run detection every single frame, and to prevent counting failure on noisy detection.

## Running
To run, make sure to have all the dependencies installed:
- numpy
- opencv-python (minimum version 3.6.4)
- tensorflow
- numba
- scipy
- filterpy
- matplotlib

and have a directory called samples, with this video :

[Traffic Video](https://www.youtube.com/watch?v=wqctLW0Hb_0)

and rename it to traffic.mp4.

TODO : 
- cli arg to supply own video
- improve detection + tracking
