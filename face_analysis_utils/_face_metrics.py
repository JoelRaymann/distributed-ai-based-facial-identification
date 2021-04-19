import numpy as np
import math
from PIL import Image

# Metrics
def euclidean_distance(feature_1: np.ndarray, feature_2: np.ndarray) -> float:
    """
    Function to find the euclidean distance between feature vector 1 and 
    feature vector 2

    Args:
        feature_1 (np.ndarray): The first feature vector input
        feature_2 (np.ndarray): The second feature vector input

    Returns:
        float: The euclidean distance between the feature vectors
    """
    assert (feature_1.shape == feature_2.shape), f"""
    [ERROR]: Feature vectors shape mismatch! Feature vectors must have the same shape
    [ERROR]: Expected {feature_1.shape} - got {feature_2.shape}
    """

    distance = feature_1 - feature_2
    distance = np.sum(np.multiply(distance, distance))
    return np.sqrt(distance)

def cosine_distance(feature_1: np.ndarray, feature_2: np.ndarray) -> float:
    """
    Function to find the cosine similarity between the given feature vector 1 and
    feature vector 2

    Args:
        feature_1 (np.ndarray): The given input feature vector 1
        feature_2 (np.ndarray): The given input feature vector 2

    Returns:
        float: The calculated euclidean distance.
    """
    feature_1 = np.squeeze(feature_1)
    feature_2 = np.squeeze(feature_2)
    assert (feature_1.shape == feature_2.shape), f"""
    [ERROR]: Feature vectors shape mismatch! Feature vectors must have the same shape
    [ERROR]: Expected {feature_1.shape} - got {feature_2.shape}
    """

    assert (len(feature_1.shape) == 1), f"""
    [ERROR]: Expected 1-dimensional vector for feature_1 - got {feature_1.shape} after squeeze!
    """

    assert (len(feature_2.shape) == 1), f"""
    [ERROR]: Expected 1-dimensional vector for feature_2 - got {feature_2.shape} after squeeze!
    """

    a = np.matmul(np.transpose(feature_1), feature_2)
    b = np.sum(np.multiply(feature_1, feature_1))
    c = np.sum(np.multiply(feature_2, feature_2))

    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))




