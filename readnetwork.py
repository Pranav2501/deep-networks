# CS 5330 - Computer Vision
# Project 5
# Name: Pranav Raj Sowrirajan Balaji
# Date: 26 March 2024
# Purpose : To read a trained network and test it on the first 10 examples
# Tasks: 
# 1. Build and train a network to recognize digits
# E. Read the network and test it on the first 10 examples

# Import the required libraries
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

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



    # Load the test data
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/Users/pranavraj/Desktop/visualstudio-opencv/files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=10, shuffle=False)

    # Get the first batch of test data
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    # Run the model on the first 10 examples
    output = network(example_data)

    # Print the network output values, the index of the max output value, and the correct label for each example
    for i in range(10):
        print("Network output: ", output.data[i].numpy().round(2))
        print("Max output value index: ", output.data.max(1, keepdim=True)[1][i].item())
        print("Correct label: ", example_targets[i].item())
        print()

    # Plot the first 9 digits with the prediction for each example
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

if __name__ == "__main__":
    main(sys.argv)