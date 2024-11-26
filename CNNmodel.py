#Emotion Classification Model


#__________________________________ Step 1: Setting Up the Environment _______________________________________________

#Install Dependencies
#pip install torch torchvision matplotlib Pillow kaggle


#__________________________________ Step 2: Import Required Libraries _______________________________________________


# PyTorch has two primitives to work with data: torch.utils.data.DataLoader and torch.utils.data.Dataset. 
# Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset.
# PyTorch offers domain-specific libraries such as TorchText, TorchVision, and TorchAudio, all of which include datasets. 

import os
import zipfile
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from kaggle.api.kaggle_api_extended import KaggleApi


# Function to download FER2013 dataset
def download_fer2013_dataset():
    # Define the path where you want to download the dataset
    dataset_dir = './data/fer2013'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Authenticate using the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset from Kaggle
    api.dataset_download_files('jonathanoheix/face-expression-recognition-dataset', path=dataset_dir, unzip=True)

    print("Dataset downloaded and unzipped!")
    
# Step 1: Download the FER2013 dataset (if not already downloaded)
download_fer2013_dataset()


#_________________________ Step 3: Load the FER2013 Dataset Using torchvision.datasets _______________________________

# The torchvision.datasets module contains Dataset objects for many real-world vision data like FER2013
# Every TorchVision Dataset includes two arguments: 
# transform and target_transform to modify the samples and labels respectively.

# Define the transformations: resizing, converting to grayscale, and normalizing
# ensures that the images are resized, converted to grayscale (if not already), and normalized.
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure it's grayscale
    transforms.Resize((48, 48)),  # Resize to 48x48 pixels
    transforms.ToTensor(),  # Convert to tensor (values between 0 and 1)
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the pixel values
])

# Download training data from open datasets
# train_data = datasets.FER2013(
#     root="data",
#     train=True,
#     transform=transform,
# )

train_data = datasets.FER2013(
    root='./data/fer2013',  # Path where the dataset was downloaded
    split='train',  # 'train' or 'test'
    transform=transform
)

# test_data = datasets.FER2013(
#     root="data",
#     train=False,
#     transform=transform,
# )

test_data = datasets.FER2013(
    root='./data/fer2013',  # Path where the dataset was downloaded
    split='test',  # 'train' or 'test'
    transform=transform
)


# Create DataLoader for batching and shuffling
batch_size = 64
# training_loader = DataLoader(train_data, batch_size=batch_size)
# test_loader = DataLoader(test_data, batch_size=batch_size)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# We pass the Dataset as an argument to DataLoader. This wraps an iterable over our dataset, and supports automatic batching, 
# sampling, shuffling and multiprocess data loading. Here we define a batch size of 32, 
# i.e. each element in the dataloader iterable will return a batch of 32 features and labels.

for X, y in test_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


#___________________________________________ Step 4: Define the CNN Model _________________________________________________


# To define a neural network in PyTorch, we create a class that inherits from nn.Module. 
# We define the layers of the network in the __init__ function and specify how data will pass through the network in the forward function. 
# To accelerate operations in the neural network, we move it to the GPU or MPS if available.

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        # First convolutional layer: 1 input channel (grayscale image), 32 output channels (filters)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Max-pooling layer: downsample images by a factor of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers (after flattening the output from conv layers)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # Adjust dimensions based on image size after pooling
        self.fc2 = nn.Linear(128, 7)  # 7 emotions in FER2013: happy, sad, etc.
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # For classification output
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Apply conv1, ReLU, and pool
        x = self.pool(self.relu(self.conv2(x)))  # Apply conv2, ReLU, and pool
        x = x.view(-1, 64 * 12 * 12)  # Flatten the output
        x = self.relu(self.fc1(x))  # Apply fully connected layer 1
        x = self.fc2(x)  # Output layer (no activation for raw scores)
        return self.softmax(x)  # Softmax to get probabilities for each class


model = EmotionCNN().to(device)
print(model)


#________________________________ Step 5: Initialize the Model, Loss Function, and Optimizer ________________________________

#To train a model, we need a loss function and an optimizer.
# Initialize the model, loss function, and optimizer
model = EmotionCNN().to('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# We’re using the Adam optimizer, which is often effective for deep learning tasks.


#___________________________________________ Step 6: Train the Model ___________________________________________


# We run the training loop for a specified number of epochs (10 in this case).
num_epochs = 5

#In a single training loop, the model makes predictions on the training dataset (fed to it in batches), 
# and backpropagates the prediction error to adjust the model’s parameters.
'''
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
'''

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    # For each batch, 
    # we perform the forward pass, compute the loss, perform the backward pass, and update the weights.
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        #Compute prediction error
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute the loss
        
        #Backpropogation
        loss.backward()  # Backward pass (compute gradients)
        optimizer.zero_grad()  # Zero the gradients
        optimizer.step()  # Update weights
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    


#_________________________________________ Step 7: Evaluate the Model _________________________________________

# In evaluation mode, we don't need to track gradients, so we use torch.no_grad()
# We compute the predicted labels and compare them to the true labels to calculate the accuracy..

# Evaluation Loop: Accumulate results and compute accuracy at the end.
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

#We also check the model’s performance against the test dataset to ensure it is learning.
'''
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
'''

with torch.no_grad():  # No need to compute gradients during evaluation
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        total += targets.size(0)  # Accumulate the number of targets (total samples)
        correct += (predicted == targets).sum().item()  # Accumulate correct predictions

# After processing all batches, calculate accuracy
accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')



# The training process is conducted over several iterations (epochs). During each epoch, the model learns parameters to make better predictions. 
# We print the model’s accuracy and loss at each epoch; we’d like to see the accuracy increase and the loss decrease with every epoch.
'''
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
'''

#_________________________________________ Step 8: Save and Load the Model _________________________________________

# Save the model
torch.save(model.state_dict(), 'emotion_cnn.pth')

# To load the model later
model = EmotionCNN()
model.load_state_dict(torch.load('emotion_cnn.pth'))
model.eval()  # Set to evaluation mode


#_______________________________________________ Step 9: Visualization _______________________________________________

# Plot some examples from the test set
data_iter = iter(test_loader)
images, labels = data_iter.next()

# Show the first image and its predicted class
img = images[0].cpu().numpy().transpose((1, 2, 0))  # Convert from tensor to numpy array
plt.imshow(img.squeeze(), cmap='gray')
plt.title(f"Predicted: {labels[0].item()}")
plt.show()


# Experiment with data augmentation to improve the model's generalization.
# Tune hyperparameters like the learning rate or batch size for better performance.
# If needed, use a pre-trained model like VGG or ResNet for improved accuracy.