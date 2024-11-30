import torch
import torchvision
from torchvision import models, transforms
from PIL import Image
import os
import csv

class vehicle_classification:
    def __init__(self, debug=False):
        self.debug = debug

        self.model_path = 'resnet50_finetuned.pth'  # Path to the fine-tuned model
        # Load the fine-tuned model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the model architecture and the saved weights
        self.model = models.resnet50(pretrained=False)  # Don't load pre-trained weights since we're loading the fine-tuned one
        num_classes = 196  # Number of classes in Stanford Cars dataset
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)  # Update the final layer for the Stanford Cars dataset
        self.model.load_state_dict(torch.load(self.model_path))
        self.model = self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

        # Define the same image transformation that was used during training
        self.transform = transforms.Compose([
            transforms.Resize(256),  # Resize the image to a 256px size
            transforms.CenterCrop(224),  # Crop the image to 224x224px
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
        ])

        # Load the class labels (assuming they were saved in the same order as during training)
        #class_names = os.listdir('path/to/stanford-cars/dataset/train')  # List the class names (folders) in the 'train' directory
        self.class_names = []
        with open('stanford_cars_dataset/stanford_cars/stanford_cars_with_class_names_train.csv', 'r') as file:
            reader = csv.reader(file)
            header=True
            for row in reader:
                #print(row[6])
                if header==False:
                    self.class_names.append(row[6])
                header=False




        transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        }


        train_dataset = torchvision.datasets.StanfordCars(root='./stanford_cars_dataset', transform=transform['train'], download=False)
        self.class_names = train_dataset.classes

    def classify_vehicles(self, img):        
        # Load the image
        image = Image.open(img)

        # Apply the transformations to the image
        image_tensor = self.transform(image).unsqueeze(0)  # Add a batch dimension (unsqueeze(0))

        # Move the image tensor to the same device as the model (GPU or CPU)
        image_tensor = image_tensor.to(self.device)

        # Make the prediction
        with torch.no_grad():  # No need to compute gradients during inference
            outputs = self.model(image_tensor)
            _, predicted_class = torch.max(outputs, 1)  # Get the index of the class with the highest score

        # Get the name of the predicted class
        predicted_class_name = self.class_names[predicted_class.item()]

        # Print the result
        print(f"Predicted class: {predicted_class_name}")
        print(f"predicted_class.item(): {predicted_class.item()}")
        #print(f"class_names[0]: {class_names[0]}")
        #print(f"class_names[1]: {class_names[1]}")
        #print(f"class_names[2]: {class_names[2]}")
