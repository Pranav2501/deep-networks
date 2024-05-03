# CS 5330 - Computer Vision
# Project 5
# Name: Pranav Raj Sowrirajan Balaji
# Date: 26 March 2024
# Purpose : To use FashionMNIST dataset to train a neural network using PyTorch and evaluate different parameters to improve the model
# Tasks: 
# 4. Design your own Experiment
# A. Develop a plan B. Predict the results C. Evaluate your plan 

import torch
import sys
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from itertools import product


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Define training and testing functions
def train(dataloader, model, loss_fn, optimizer):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
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

# Define test function
def test(dataloader, model, loss_fn):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
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
    return correct, test_loss

# Define main function
def main(argv):

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    # Define batch size
    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


    # Train the model
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
    # Save the model
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
    # Load the model
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth"))

    # Evaluate the model
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

    # Define the range of options for each dimension
    epochs_options = [1, 2, 3, 4]
    batch_size_options = [5, 10, 15, 20, 25]
    activation_functions = [nn.ReLU, nn.Sigmoid, nn.Tanh]

    # Initialize an empty list to store the results
    results = []

    # Iterate over each combination of parameters
    for epochs, batch_size, activation_function in product(epochs_options, batch_size_options, activation_functions):
        # Update the model's activation function
        model.linear_relu_stack[1] = activation_function()
        model.linear_relu_stack[3] = activation_function()

        # Update the data loaders
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        # Train the model
        for t in range(epochs):
            train(train_dataloader, model, loss_fn, optimizer)

        # Evaluate the model
        correct, test_loss = test(test_dataloader, model, loss_fn)

        # Store the results
        results.append({
            'epochs': epochs,
            'batch_size': batch_size,
            'activation_function': activation_function.__name__,
            'accuracy': correct,
            'loss': test_loss
        })
        # Print the results
        print(results)

if __name__ == "__main__":
    main(sys.argv)