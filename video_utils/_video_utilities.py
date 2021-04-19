import numpy as np
import cv2.cv2 as cv
from tqdm import tqdm
import traceback

def video_writer(input_path:str, output_path: str, outputDict: dict, output_frame_resolution = (854, 480)):

    cap = cv.VideoCapture(input_path)
    frames_processed = 0

    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    prediction_out = cv.VideoWriter(output_path, fourcc, 30, output_frame_resolution)

    with tqdm(total = float("inf")) as pbar:
        while cap.isOpened():
            try:
                ret, frame = cap.read()

                # Check for proper read
                if not ret:
                    break

                # Process the frame
                frame = cv.resize(frame, output_frame_resolution, interpolation = cv.INTER_CUBIC)

                # frames processed
                frames_processed += 1
                out = outputDict.get(frames_processed, -1)
                if out != -1:
                    x, y, width, height = out["bounding_box"]
                    distance = out["distance"]
                    cv.rectangle(frame, (x, y), (x+width, y+height), (36, 255, 12), 2)
                    cv.putText(frame, 
                            f'Subject-{distance:.4f}',
                            (x, y-10),
                            cv.FONT_HERSHEY_PLAIN,
                            1.0,
                            (36, 255, 12), 
                            2)
                prediction_out.write(frame)
                cv.imshow('frame',frame)
                
                pbar.set_postfix_str(f"Frames Processed: {frames_processed}")
                pbar.update(1)

                # Break condition
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                
                
            except Exception as err:
                print(f"[ERROR]: {err}")
                print(f"[ERROR]: {traceback.print_exc()}")
                return False

    cap.release()
    prediction_out.release()
    cv.destroyAllWindows()
    return True

