from torchvision.transforms.functional import Tensor
from torchvision.transforms.functional import crop
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

class box:
    def __init__(self, xmin=0, ymin=0, xmax=9999, ymax=9999):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

class roi_extractor:
    def __init__(self, entrance, exit, debug=False):
        self.debug = debug
        self.entrance = entrance
        self.exit = exit

    def get(self, img, boxes):
        
        entrance_object = Tensor()
        exit_object = Tensor()
        for box in boxes:
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            if (xmin > self.entrance.xmin and xmax < self.entrance.xmax and
                ymin > self.entrance.ymin and ymax < self.entrance.ymax):
                height = ymax - ymin
                width = xmax - xmin
                entrance_object = crop(img, ymin, xmin, height, width)
                if self.debug:
                    print(f"Entrance object discovered at (xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

            if (xmin > self.exit.xmin and xmax < self.exit.xmax and
                ymin > self.exit.ymin and ymax < self.exit.ymax):
                height = ymax - ymin
                width = xmax - xmin
                exit_object = crop(img, ymin, xmin, height, width)
                if self.debug:
                    print(f"Exit object discovered at (xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

        if self.debug:
            self._plot(img, entrance_object, exit_object)
            
        return entrance_object, exit_object

    def _plot(self, img, entrance_object, exit_object):
        plt.subplot(3, 1, 1)
        plt.imshow(to_pil_image(img))
        plt.title('Original input image')
        
        if entrance_object.numel():
            plt.subplot(3,1 , 2)
            plt.imshow(to_pil_image(entrance_object))
            plt.title('Object detected at parking lot entrance')
        
        if exit_object.numel():
            plt.subplot(3,1 , 3)
            plt.imshow(to_pil_image(exit_object))
            plt.title('Object detected at parking lot exit')

        plt.tight_layout()
        plt.show()
