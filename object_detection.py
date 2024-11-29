from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

class object_detection:
    def __init__(self, debug=False):
        self.debug = debug

        # Initialize model with the best available weights
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=0.9)
        self.model.eval()
        # Step 2: Initialize the inference transforms
        self.preprocess = self.weights.transforms()
        
    def _preprocess(self, img):
        # Step 3: Apply inference preprocessing transforms
        self.batch = [self.preprocess(img)]

    def _predict(self):
        self.prediction = self.model(self.batch)[0]
        print (self.prediction["boxes"][0]) # Prints out xmin,ymin,xmax,ymax of detected box 0

    def _postprocess(self, img):
        labels = [self.weights.meta["categories"][i] for i in self.prediction["labels"]]
        
        if self.debug:
            box = draw_bounding_boxes(img, boxes=self.prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=4, font_size=30)
            im = to_pil_image(box.detach())
            plt.imshow(im)
            plt.show()

        return self.prediction["boxes"]

    def run(self, img):
        self._preprocess(img)
        self._predict()
        boxes = self._postprocess(img)
        return boxes

