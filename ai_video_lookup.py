# Import packages
import argparse
import yaml
import sys

# Import in-house packages
import driver_utils as du

DESCRIPTION = """
Distributed AI based Video Face Querying

A fully distributed AI powered video face querying software developed on
Apache Spark, Apache Hadoop, YARN, OpenCV and TensorFlow
"""

VERSION = "0.1-alpha"

# Main program
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = DESCRIPTION)

    # Arguments
    parser.add_argument("-V", "--version", help = "shows the program version", action="store_true")
    parser.add_argument("-I", "--input-path", help = "The input video to query", type = str)
    parser.add_argument("-O", "--output-path", help = "The otuput path to dump the resulting video", type = str)
    parser.add_argument("-Q", "--query-path", help = "The path to the query image face", type = str)
    parser.add_argument("-C", "--config-path", help = "The path to the YAML Configuration file", type = str)
    parser.add_argument("--full-spark", help = "Flag to indicate the program to use Full Spark Architecture instead of Hybrid", default = False, type = bool)

    args = parser.parse_args()

    # Check for version
    if args.version:
        print(f"[INFO]: Using Version {VERSION}")
        sys.exit(1)
    
    input_path = args.input_path if args.input_path else None
    output_path = args.output_path if args.output_path else None
    config_path = args.config_path if args.config_path else "configuration.yaml"
    query_path = args.query_path if args.query_path else None
    execution_option = True if args.full_spark else False

    if input_path is None or output_path is None or query_path is None:
        print("[ERROR]: Missing Arguments")
        sys.exit(0)
    
    print("[INFO]: Parsing YAML Configuration File")
    config = yaml.load(open(config_path, mode = "r"), Loader = yaml.FullLoader)

    print("[INFO]: The following configuration is parsed")
    for k, v in config.items():
        print("{0}: {1}".format(k, v))
    
    if execution_option:
        # TODO: Full Spark Program
        print("[ERROR]: Not Implemented Full Spark")
        raise NotImplementedError()
    
    else:
        if du.run_local_query(input_path, output_path, query_path, config):
            print(f"[SUCCESS]: Program completed and output stored in {output_path}")
        else:
            print("[FAILURE]: Program failed")
