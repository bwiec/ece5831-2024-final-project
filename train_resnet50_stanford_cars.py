import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
import os

# Define the path to the Stanford Cars dataset
data_dir = 'stanford_cars_dataset/stanford_cars'

# Define image transformations for training and validation
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

# Load the Stanford Cars dataset (Assuming you have organized the dataset in ImageFolder format)
#train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform['train'])
#val_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform['val'])

#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#bw
train_dataset = torchvision.datasets.StanfordCars(root='./stanford_cars_dataset', transform=transform['train'], download=False)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=1)

val_dataset = torchvision.datasets.StanfordCars(root='./stanford_cars_dataset', transform=transform['val'], download=False)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=1)


# Load a pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Freeze all layers except the final layer
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer to match the number of classes in Stanford Cars
num_classes = len(train_dataset.classes)  # Number of classes in Stanford Cars
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Send model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training loop
num_epochs = 2

for epoch in range(num_epochs):
    print("Starting training")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_accuracy:.2f}%\n")

# Save the fine-tuned model
torch.save(model.state_dict(), 'resnet50_finetuned.pth')
