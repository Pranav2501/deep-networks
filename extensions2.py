# CS 5330 - Computer Vision
# Project 5
# Name: Pranav Raj Sowrirajan Balaji
# Date: 26 March 2024
# Purpose : To use fashion MNIST dataset to train a network and classify live frames from the webcam
# Tasks: 
# 1. Extensions: The script performs image classification using a pre-trained  network
# Dataset: Fashion MNIST 






import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import torch.nn.functional as F
import sys


# Define the network architecture
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def main(argv):
    # Load and preprocess the dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Create the network, define the criterion and optimizer
    model = Classifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    for epoch in range(5):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Save the model
    torch.save(model.state_dict(), 'fashion_mnist_model.pth')

    # Load the saved model weights
    model.load_state_dict(torch.load('fashion_mnist_model.pth'))
    model.eval()  # Set the model to evaluation mode

    # Function to preprocess the live frames
    def preprocess_frame(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (28, 28))
        frame = frame / 255.0
        frame = frame.reshape(1, 1, 28, 28)
        frame = torch.from_numpy(frame).float()
        return frame

    # Capture live frames from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Webcam', frame)

        class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        key = cv2.waitKey(1)
        if key == ord('s'):
            frame = preprocess_frame(frame)
            prediction = model(frame)
            predicted_class_index = torch.argmax(prediction).item()
            predicted_class_label = class_labels[predicted_class_index]
            print('Predicted class:', predicted_class_label)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)