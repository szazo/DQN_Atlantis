"""
import cv2
import numpy as np


def resize_frame(frame):
    frame = frame[28:,5:-4]
    frame = np.average(frame,axis = 2)
    frame = cv2.resize(frame,(84,84),interpolation = cv2.INTER_NEAREST)
    frame = np.array(frame,dtype = np.uint8)
    return frame
"""

import cv2
import numpy as np


def resize_frame(frame):
    frame = frame[0:-30]
    frame = np.average(frame,axis = 2)

    target = frame
#    target = np.zeros((170, 170)) # zero padding
#    target[:,5:-5] = frame

    #target = cv2.resize(target,(90,80),interpolation = cv2.INTER_NEAREST)
    target = np.array(target,dtype = np.uint8)
    return target
