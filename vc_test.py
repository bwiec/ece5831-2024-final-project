from torchvision.io.image import decode_image
from vehicle_classification import vehicle_classification
from PIL import Image

dbg = True

#img = decode_image("stanford_cars_dataset/stanford_cars/cars_test/00007.jpg")
file = "stanford_cars_dataset/stanford_cars/cars_test/0000"

vc = vehicle_classification(debug=dbg)

for ii in range(1,9):
    img = Image.open(file + str(ii) + '.jpg')
    vehicle_class = vc.classify_vehicles(img)


