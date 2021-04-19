# Import the necessary packages
import numpy as np
import pyspark as spark
import tensorflow as tf

# Import in-house-packages
import local_utils as lu
import spark_utils as su
import video_utils as vu

# Run locally
def run_local_query(input_path: str, output_path: str, query_image_path: str, config: dict):

    print("[INFO]: Starting a Spark-GPU Cluster based Lookup")
    
    print("[INFO]: Optimizing GPU-Cluster")
    # GPU Optimization
    try:
        physical_devices = tf.config.list_physical_devices("GPU")
        for index in range(len(physical_devices)):
            try:
                tf.config.experimental.set_memory_growth(physical_devices[index], True)

            except Exception as err:
                print("[WARN]: Failed to set memory growth for {0}".format(physical_devices[index]))
                print("[WARN]: Error", err, " .Skipping memory optimization")

    except Exception as err:
        print("[WARN]: memory optimization failed. Error:", err, " . Skipping!")
    
    # YAML
    frame_resolution = (int(config["frame_resolution"]["height"]), 
                        int(config["frame_resolution"]["width"]))
    mtcnn_threshold = int(config["mtcnn_threshold"])

    # Get the frames from the video
    frames = lu.get_frames_locally(input_path, frame_resolution)

    # Load the AI Model
    model = tf.keras.models.load_model("./model/face_recog.h5")

    # Get the faces
    model_resolution = model.input_shape[1:-1]
    face_output = lu.get_faces_locally(frames, model_resolution)

    # Get the Face Embedding
    emb_output = lu.get_face_embeddings_locally(face_output, model)

    # Get the query vector
    _, query_vector = lu.input_preprocessing(query_image_path, model)

    # Run the face verification
    return _run_verification_parallel(emb_output, query_vector, input_path, output_path, config)
    


def _run_verification_parallel(emb_output, query_vector, input_path: str, output_path: str, config: dict):

    # YAML
    frame_resolution = (int(config["frame_resolution"]["height"]), 
                        int(config["frame_resolution"]["width"]))
    face_threshold = float(config["face_threshold"])
    master = config["master"]

    # Set the Spark Context
    conf = spark.SparkConf().setAppName("video_query").setMaster("yarn")
    sc = spark.SparkContext(conf = conf)
    print(sc)

    # Add the spark modules
    sc.addPyFile("./modules/face_analysis_utils.zip")
    sc.addPyFile("./modules/spark_utils.zip")

    # Start the spark
    inputRDD = sc.parallelize(emb_output)
    
    # Process the spark
    outputRDD = inputRDD.map(lambda x: su.face_verification(x[0], x[1], query_vector, threshold=face_threshold))
    output = outputRDD.filter(lambda x: x is not None).sortByKey(True, numPartitions = 1).collect()

    # Convert the dictionary
    outputDict = dict()
    for k, v in output:
        out = outputDict.get(k, -1)
        if out == -1:
            outputDict[k] = v
        elif out["distance"] > v["distance"]:
            outputDict[k] = v
        else:
            continue
    
    # Write the output
    return vu.video_writer(input_path, output_path, outputDict, frame_resolution)




    



