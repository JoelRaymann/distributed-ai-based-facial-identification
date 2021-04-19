# Import the packages
import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv
import tensorflow as tf
from PIL import Image
import pyspark as spark
import io
from mtcnn_cv2 import MTCNN

## Utility Function
def get_faces(binary_image_bytes, threshold = 0.9) -> tuple:

    # Import the stuffs
    import face_analysis_utils as fau

    # Get the meta
    image_name, binary_image = binary_image_bytes

    frame_no = int(image_name.split("$")[-1].split(".")[0])

    # Decode the image frame
    img = Image.open(io.BytesIO(binary_image))
    img = np.asarray(img, dtype=np.uint8)
    if img.shape[-1] == 4:
        img = img[:, :, :-1].copy()
    else:
        img = img[:, :, :].copy()
    
    # Load the MTCNN model
    detector = MTCNN()
    faces = detector.detect_faces(img)

    if len(faces) > 0:

        for index, face in enumerate(faces):
            if face["confidence"] < threshold:
                yield None
            else:
                # Snip the image
                x, y, width, height = face["box"]
                face_img = img[y:y+height, x:x+width]
                
                # Face Alignment
                key = face["keypoints"]
                face_img = fau.face_alignment(face_img,
                                              key["left_eye"],
                                              key["right_eye"])
                
                # Resize the image
                face_img = cv.resize(face_img, 
                                     (112, 112),
                                     interpolation = cv.INTER_CUBIC)
                face_img = np.asarray(face_img / 255., dtype="float64")
                
                # return the snipped face with bounding box
                yield (frame_no, {
                    "bounding_box": face["box"],
                    "face_image": face_img
                })
    else:
        return None

def run_ai_model_gpu(model: tf.keras.Model, input_face_images: np.ndarray):

    if len(input_face_images) == 3:
        input_face_images = input_face_images[None, ...]
    
    face_embeddings = model(input_face_images)
    return face_embeddings


def get_face_embeddings(binary_image_bytes, threshold = 0.9) -> tuple:
    """
    Function to detect the faces using MTCNN and run face embedding
    model using the ArcFace model

    Args:
        binary_image_bytes (): The binary image format
        threshold (float, optional): The threshold to prune out wrong faces. Defaults to 0.9.

    Returns:
        tuple: The tuple of (str, dict)
    """
    # Import the stuffs
    import face_analysis_utils as fau

    # Get the meta
    image_name, binary_image = binary_image_bytes

    frame_no = int(image_name.split("$")[-1].split(".")[0])

    # Decode the image frame
    img = Image.open(io.BytesIO(binary_image))
    img = np.asarray(img, dtype=np.uint8)
    if img.shape[-1] == 4:
        img = img[:, :, :-1].copy()
    else:
        img = img[:, :, :].copy()
    
    # Load the MTCNN model
    detector = MTCNN()
    faces = detector.detect_faces(img)
    
    if len(faces) > 0:
        
        # Load the face-recognition model
        face_images = []
        bounding_boxes = []
        for index, face in enumerate(faces):
            # First, prune out low confident predictions
            if face["confidence"] < threshold:
                yield None
            else:
                # Snip the image
                x, y, width, height = face["box"]
                face_img = img[y:y+height, x:x+width]
                
                # Face Alignment
                key = face["keypoints"]
                face_img = fau.face_alignment(face_img,
                                              key["left_eye"],
                                              key["right_eye"])
                
                # Resize the image
                face_img = cv.resize(face_img, 
                                     (112, 112),
                                     interpolation = cv.INTER_CUBIC)
                face_img = np.asarray(face_img / 255., dtype="float64")
                
                # Add the faces
                face_images.append(face_img[None, ...])
                bounding_boxes.append(face["box"])
            
        # Do Face Embedding
        if len(face_images) > 0:
            face_images = np.concatenate(face_images, axis = 0) if len(face_images) > 1 else face_images[0]
            
            # Get the embeddings
            # Running in CPU to make it Robust
            # Using GPU requires CUDA built Spark
            with tf.device("CPU"):
                path = spark.SparkFiles.get("face_recog.h5")
                model = tf.keras.models.load_model(path)
                
                # Get the embeddings
                embeddings = model(face_images)
            
            # Now get the numpy version
            for index, embedding in enumerate(embeddings.numpy()):
                
                # Construct the info dictionary
                output = {
                    "bounding_box": bounding_boxes[index],
                    "embedding": embedding[None, ...]
                }
                
                yield (frame_no, output)
        else:
            yield None
    else:
        yield None

def face_verification(frame_no: int, face_data: dict, query_face_vector: np.ndarray, metric = "cosine", threshold = 0.6871912) -> tuple:
    """
    Function to do the face verification and prune out frames that does not have the 
    query face.

    Args:
        frame_no (int): The frame number being processed.
        face_data (dict): A dictionary consisting of the bounding box 
        and the face embeddings
        query_face_vector (np.ndarray): The query face vector.
        metric (str, optional): The metric to use for confirmation. Defaults to "cosine".
        threshold (float, optional): The threshold to finalize the face verification. Defaults to 0.6871912.

    Returns:
        tuple: The output frames with the distance value.
    """
    # Import the stuffs
    import face_analysis_utils as fau

    face_vector = face_data["embedding"]
    bounding_box = face_data["bounding_box"]
    # Find the distance
    if metric == "cosine":
        distance = fau.cosine_distance(face_vector, query_face_vector)
    elif metric == "euclidean":
        distance = fau.euclidean_distance(face_vector, query_face_vector)
    else:
        raise AssertionError("[ERROR]: Metric Not Supported!")
    
    if distance <= threshold:
        return (frame_no, {
            "bounding_box": bounding_box,
            "distance": distance
        })
    else:
        return None

    



    
    