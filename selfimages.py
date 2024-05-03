# CS 5330 - Computer Vision
# Project 5
# Name: Pranav Raj Sowrirajan Balaji
# Date: 26 March 2024
# Purpose : To use the trained network to recognize custom images
# Tasks: 
# 1. Build and train a network to recognize digits
# F. Test the network on new images




from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import sys

# Define the Net class for the neural network
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


    # List of image paths
    image_paths = ['/Users/pranavraj/Desktop/visualstudio-opencv/Mydigits/IMG_1644.jpg', '/Users/pranavraj/Desktop/visualstudio-opencv/Mydigits/Cropped_IMG_1633.jpg', '/Users/pranavraj/Desktop/visualstudio-opencv/Mydigits/Cropped_IMG_1634.jpg', '/Users/pranavraj/Desktop/visualstudio-opencv/Mydigits/IMG_1635.jpg', '/Users/pranavraj/Desktop/visualstudio-opencv/Mydigits/IMG_1636.jpg','/Users/pranavraj/Desktop/visualstudio-opencv/Mydigits/IMG_1638.jpg','/Users/pranavraj/Desktop/visualstudio-opencv/Mydigits/IMG_1639.jpg','/Users/pranavraj/Desktop/visualstudio-opencv/Mydigits/IMG_1640.jpg','/Users/pranavraj/Desktop/visualstudio-opencv/Mydigits/IMG_1641.jpg']

    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Process the images and run them through the network
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))  # Create a 3x3 grid of subplots
    axs = axs.ravel()  # Flatten the grid to easily iterate over it

    for i, image_path in enumerate(image_paths):
        # Open the image file
        img = Image.open(image_path)

        # Rotate the image
        img = img.rotate(270)  # replace 'angle' with the angle you want to rotate the image

        # Invert the colors
        img = ImageOps.invert(img)
        
        # Apply the transforms
        img_t = transform(img)
        
        # Add an extra batch dimension
        img_t_unsqueezed = img_t.unsqueeze(0)
        
        # Run the network on the image
        output = network(img_t_unsqueezed)
        
        # Get the predicted digit
        predicted_digit = output.data.max(1, keepdim=True)[1].item()
        
        # Display the image along with the prediction in the subplot
        axs[i].imshow(img_t[0], cmap='gray')
        axs[i].set_title(f"Predicted digit: {predicted_digit}")
        axs[i].axis('off')  # Hide the axis

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)