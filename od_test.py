from torchvision.io.image import decode_image
from object_detection import object_detection
from roi_extractor import roi_extractor

dbg = True

img = decode_image("data/img1.jpg")
od = object_detection(debug=False)
boxes = od.run(img)

ri = roi_extractor(debug=dbg)
rois = ri.get(img, boxes)


