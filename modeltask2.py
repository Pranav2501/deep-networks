# CS 5330 - Computer Vision
# Project 5
# Name: Pranav Raj Sowrirajan Balaji
# Date: 26 March 2024
# Purpose : To analyze the network and show the effects of the filters on images
# Tasks: 
# 2. Examine the network
# A. Analyze the first layer filters & B. Show the effects of the filters on images

# Import the required libraries
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision  # Add this line
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Define the Net class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

# Define the main function

def main(argv):
    # Load the trained network
    network = Net()
    network.load_state_dict(torch.load('/Users/pranavraj/Desktop/visualstudio-opencv/results/model.pth'))
    network.eval()

    # Print the structure of the network
    print(network)

    # Get the weights of the first layer
    weights = network.conv1.weight.detach()

    # Print the filter weights and their shape
    for i in range(10):
        print("Filter", i+1)
        print(weights[i, 0])
        print("Shape:", weights[i, 0].shape)
        print()

    # Visualize the ten filters
    fig = plt.figure()
    for i in range(10):
        plt.subplot(3, 4, i+1)
        plt.tight_layout()
        plt.imshow(weights[i, 0], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # Load the first training example image
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/Users/pranavraj/Desktop/visualstudio-opencv/files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=1, shuffle=False)
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    image = example_data[0][0].numpy()

    # Apply the filters to the image
    filtered_images = []
    with torch.no_grad():
        for i in range(10):
            filter = weights[i, 0].numpy()
            filtered_image = cv2.filter2D(image, -1, filter)
            filtered_images.append(filtered_image)

    # Plot the filtered images
    fig = plt.figure()
    for i in range(10):
        plt.subplot(3, 4, i+1)
        plt.tight_layout()
        plt.imshow(filtered_images[i], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()

if __name__ == "__main__":
    main(sys.argv)