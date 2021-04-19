import numpy as np
import cv2.cv2 as cv
import mtcnn_cv2
from PIL import Image
import math

from ._face_metrics import euclidean_distance


def face_alignment(face_image: np.ndarray, left_eye: tuple, right_eye: tuple) -> np.ndarray:
    """
    Function to align the given face image appropriately following the procedures
    laid in the ArcFace and Google FaceNet paper.
    NOTE: This function is developed by @serengil under the MIT License.

    Args:
        face_image (np.ndarray): The cropped image of the face
        left_eye (float): The location of the left_eye
        right_eye (float): The location of the right_eye

    Returns:
        np.ndarray: The aligned face image
    """
    # this function aligns given face in img based on left and right eye coordinates
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    #-----------------------
    #find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock

    #-----------------------
    #find length of triangle edges

    a = euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = euclidean_distance(np.array(right_eye), np.array(left_eye))

    #-----------------------

    #apply cosine rule

    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation

        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / math.pi #radian to degree

        #-----------------------
        #rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(face_image)
        img = np.array(img.rotate(direction * angle))
    
    #-----------------
    return img
