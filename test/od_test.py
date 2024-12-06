import os
import sys
import argparse
from torchvision.io.image import decode_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from object_detection import object_detection

def setup_args():
    arg_parser = argparse.ArgumentParser(prog=sys.argv[0], description="Unit tester for object detection")
    arg_parser.add_argument('--debug', action='store_true', help='Enable debug messages/images')
    arg_parser.add_argument('files', nargs='+', help='Image file(s) to test')
    
    Args = sys.argv
    Args.pop(0)
    args = arg_parser.parse_args(Args)
    
    dbg = args.debug

    files = []
    for file in args.files:
        files.append(os.path.abspath(file))

    return dbg, files

if __name__ == "__main__":
    
    dbg, files = setup_args()
    od = object_detection(debug=dbg)

    for file in files:
        img = decode_image(file)
        boxes = od.run(img)

