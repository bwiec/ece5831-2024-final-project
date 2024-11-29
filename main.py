import argparse
import sys
from debug import debug
from parking_state import parking_state
from video_capture import video_capture
from object_detection import object_detection
from roi_extractor import roi_extractor
from vehicle_classification import vehicle_classification

def parse_args():
    arg_parser = argparse.ArgumentParser(prog=sys.argv[0], description="Final Project - Parking Spot Availability Tracker")
    arg_parser.add_argument('--debug', action='store_true', help='Enable debug messages')
    arg_parser.add_argument('file', nargs='+', help='Image file to test')


    Args = sys.argv
    Args.pop(0)
    args = arg_parser.parse_args(Args)

    return args


if __name__ == "__main__":
    
    args = parse_args

    dbg = debug(args.debug)
    state = parking_state()
    vid_capture = video_capture(args.file[0])
    obj_detect = object_detection(vid_capture.get_images())
    roi_extract = roi_extractor(vid_capture.get_images(), obj_detect.get_objects())
    vehicle_classify = vehicle_classification()

    success, frame = vid_capture.get_frame()
    while success:
        detections = obj_detect.get_detections(frame)
        roi = roi_extractor(frame, detections)
        vehicle_class = vehicle_classify.classify_vehicles(roi)
        state.update(vehicle_class)
        success, frame = vid_capture.get_frame()
