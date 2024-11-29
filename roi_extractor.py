from torchvision.transforms.functional import crop
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


class roi_extractor:
    def __init__(self, debug=False):
        self.debug = debug

    def get(self, img, boxes):
        rois = []
        for box in boxes:
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            height = ymax - ymin
            width = xmax - xmin
            rois.append(crop(img, ymin, xmin, height, width))
        
        if self.debug:
            plt.imshow(to_pil_image(rois[0]))
            plt.show()

        return rois