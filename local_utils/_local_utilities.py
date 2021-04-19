import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv
import tensorflow as tf
import io
from tqdm import tqdm
from PIL import Image
from mtcnn_cv2 import MTCNN

# Import stuffs
import face_analysis_utils as fau

def get_frames_locally(video_path: str, output_frame_resolution = (640, 360)) -> list:

    cap = cv.VideoCapture(video_path)
    frames_read = 0

    frames = []
    print("[INFO]: Getting Frames from the video")
    with tqdm(total = float("inf")) as pbar:
        while cap.isOpened():

            try:
                ret, frame = cap.read()

                # Check for proper read
                if not ret:
                    break

                # Process the frame
                frame = cv.resize(frame, output_frame_resolution, interpolation = cv.INTER_CUBIC)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frames.append(frame)

                # frames processed
                frames_read += 1
                pbar.set_postfix_str(f"Frames Processed: {frames_read}")
                pbar.update(1)
            
            except Exception as err:
                print(f"[ERROR]: {err}")
    
    print(f"""
    [INFO]: Processed {frames_read} frames from the {video_path}
    """)
            
    return frames

def get_faces_locally(images: list, output_frame_resolution = (112, 112), threshold = 0.9) -> list:

    detector = MTCNN()
    print("[INFO]: Detecting Faces in each frames")
    output = []
    for index, image in tqdm(enumerate(images)):
        if image.shape[-1] == 4:
            image = image[:, :, :-1].copy()
        
        # Detect faces
        faces = detector.detect_faces(image)

        if len(faces) > 0:

            for face in faces:
                if face["confidence"] < threshold:
                    continue
                else:
                    # Snip the image
                    x, y, width, height = face["box"]
                    face_img = image[y:y+height, x:x+width]

                    # Face alignment
                    key = face["keypoints"]
                    face_img = fau.face_alignment(face_img, key["left_eye"], key["right_eye"])

                    # Resize the image
                    face_img = cv.resize(face_img, (112, 112), interpolation = cv.INTER_CUBIC)
                    face_img = np.asarray(face_img / 255., dtype = "float64")

                    output.append((index + 1, {
                        "bounding_box": face["box"],
                        "face_image": face_img[None, ...]
                    }))
        else:
            continue
    
    if len(output) > 0:
        print(f"[INFO]: Successfully found {len(output)} faces!")
        return output
    else:
        print("[WARN]: No faces were detected at all throughout the entire series of frames")
        return None

def get_face_embeddings_locally(face_datas: list, model: tf.keras.Model) -> list:

    assert (len(face_datas) > 0), "[ERROR]: No face data given!"

    print("[INFO]: Finding the face embedding for each face")
    faces = []
    print("[INFO]: Accumulating faces...")
    for face_data in tqdm(face_datas):
        _, face = face_data
        faces.append(face["face_image"])
    
    faces = np.asarray(np.concatenate(faces, axis = 0), dtype="float64")

    print("[INFO]: Getting face embeddings")
    embeddings = model.predict(faces)
    output = [] 

    print("[INFO]: Accumulating Results")
    for index, embedding in tqdm(enumerate(embeddings)):

        frame_no, face = face_datas[index]
        output.append((frame_no, {
            "bounding_box": face["bounding_box"],
            "embedding": embedding
        }))
    print("[INFO]: Successfully got all the embeddings")
    return output


def input_preprocessing(input_image_path: str, model: tf.keras.Model) -> tuple:
    
    # Open the image
    input_image = cv.imread(input_image_path)

    # Pre-process it out
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    
    # Run MTCNN
    detector = MTCNN()
    faces = detector.detect_faces(input_image)
    
    if (len(faces) > 0):
        # Get the first face
        for face in faces:
            if face["confidence"] > 0.9:
                # Crop it out
                x, y, width, height = face["box"]
                face_image = np.asarray(input_image[y:y+height, x:x+width], dtype = np.uint8)
                
                key = face["keypoints"]
                face_image = fau.face_alignment(face_image,
                                                key["left_eye"],
                                                key["right_eye"])
                
                # Interpolate
                face_image = cv.resize(face_image, (112, 112), interpolation = cv.INTER_CUBIC)
                
                # Normalize
                face_image = np.asarray(face_image / 255.0, dtype = "float64")
                
                return face_image, model(face_image[None, ...]).numpy()
            else:
                continue
        raise Exception("[ERROR]: Proper Face not found!")
    else:
        raise Exception("[ERROR]: NO FACE FOUND IN IMAGE")

        


                