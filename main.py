import os
import sys
import argparse
from torchvision.io.image import decode_image
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from object_detection import object_detection
from roi_extractor import roi_extractor, box
from vehicle_classification import vehicle_classification
from parking_state import parking_state

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

def plot(img, entrance_object, entrance_class, exit_object, exit_class):
    plt.subplot(3, 1, 1)
    plt.imshow(to_pil_image(img))
    plt.title('Original input image')
    
    if entrance_object.numel():
        plt.subplot(3, 1 ,2)
        plt.imshow(to_pil_image(entrance_object))
        plt.title(f'Object detected at parking lot entrance ({entrance_class})')
    
    if exit_object.numel():
        plt.subplot(3, 1, 3)
        plt.imshow(to_pil_image(exit_object))
        plt.title(f'Object detected at parking lot exit ({exit_class})')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    dbg, files = setup_args()
    
    entrance = box(xmin=450, ymin=210, xmax=999, ymax=360) # Use locations for camera6
    exit = box(xmin=10, ymin=210, xmax=520, ymax=360)

    od = object_detection(debug=dbg)
    ri = roi_extractor(entrance, exit, debug=dbg)
    vc = vehicle_classification(debug=dbg)
    state = parking_state(debug=dbg)

    for file in files:
        print(f'Proccessing image {file}')

        img = decode_image(file)
        boxes = od.run(img)
        entrance_object, exit_object = ri.get(img, boxes)

        entrance_vehicle_class = []
        if entrance_object.numel():
            print("Entrance vehicle: ")
            entrance_vehicle_class = vc.classify_vehicles(to_pil_image(entrance_object))

        exit_vehicle_class = []
        if exit_object.numel():
            print("Exit vehicle: ")
            exit_vehicle_class = vc.classify_vehicles(to_pil_image(exit_object))

        if state.update(entrance_vehicle_class, exit_vehicle_class):
            state.print()
            plot(img, entrance_object, entrance_vehicle_class, exit_object, exit_vehicle_class)
        



