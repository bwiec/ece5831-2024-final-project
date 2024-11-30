from torchvision.io.image import decode_image
from object_detection import object_detection
from roi_extractor import roi_extractor
from vehicle_classification import vehicle_classification
from torchvision.transforms.functional import to_pil_image

img = decode_image("data/img1.jpg")
img = decode_image("stanford_cars_dataset/stanford_cars/cars_test/00004.jpg")
img = decode_image("cnrpark_dataset/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/SUNNY/2015-11-12/camera2/2015-11-12_1044.jpg")
od = object_detection(debug=False)
boxes = od.run(img)

ri = roi_extractor(debug=True)
rois = ri.get(img, boxes)

vc = vehicle_classification(debug=True)
vehicle_class = vc.classify_vehicles(to_pil_image(rois[0]))


