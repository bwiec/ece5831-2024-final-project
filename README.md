# Smart Parking Lot Monitoring using Image Object Detection and Classification

## Abstract
In the technologies of smart cities and autonomous vehicles, a common use case is Automated Valet Parking (AVP) where the driver will exit his or her vehicle at the destination and command the autonomous vehicle to park itself according to nearby available parking spots. The vehicle will need to communicate with infrastructure to find the nearby parking lots and learn their occupancy so it can decide where to navigate.  In this paper, we propose a novel technique for monitoring parking lot occupancy using as few as one single camera that is properly positioned. We develop machine-learning based perception algorithm that monitors the entry/exit point(s) of a parking lot and tracks the occupancy of the parking lot based on the types of available parking spots.

## Software Architecture
The software architecture is broken up into several modules:
* `object_detection.py` - Detect objects in input image
* `roi_extractor.py` - Search the detected object list and crop out region(s) of interest from the original image for any objects detected at the designated entrance/exit coordinates
* `vehicle_classification` - Fine-grained classifier to identify specific vehicle make/model
* `parking_state` - Tracks current parking lot state for various types of spots

The `main.py` module contains the main logic that puts all these modules together.

Additionally, the `test` directory contains unit tests and integration tests for demonstrating the use of individual modules.

## Running the demonstration
`final-project.ipynb` shows how to run the demonstration.

`main.py` is the main program to run and it takes one or more image files as input and processes them individually while keeping track of parking lot state

```bash
main.py --help
usage: main.py [-h] [--debug] files [files ...]

Unit tester for object detection

positional arguments:
  files       Image file(s) to test

options:
  -h, --help  show this help message and exit
  --debug     Enable debug messages/images
```

## Links
* Datasets
  * StanfordCars - https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars
  * CNRPark - https://www.kaggle.com/datasets/ddsshubham/cnrpark-ext
* Presentation - 
* Final Report - https://docs.google.com/document/d/1MW4-hHf1tfZ2vtGx_va2WhX37AJSRMKY/edit?usp=drive_link&ouid=109767501502679541558&rtpof=true&sd=true
* Demo video - 