# CS 5330 - Computer Vision
# Project 5
# Name: Pranav Raj Sowrirajan Balaji
# Date: 26 March 2024
# Purpose : To implement a neural network using PyTorch and visualize the network using torchviz
# Tasks: 
# 1. Build and train a network to recognize digits
# A. Get the dataset, B. Build the model C. Train the model D. Save it 

import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot
import sys


# Neural network class with forward method
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
# Train the network
def train(epoch, network, optimizer, train_loader, train_losses, train_counter, log_interval):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), '/Users/pranavraj/Desktop/visualstudio-opencv/results/model.pth')
      torch.save(optimizer.state_dict(), '/Users/pranavraj/Desktop/visualstudio-opencv//results/optimizer.pth')


# Test the network
def test(network, test_loader, test_losses):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

# Main function
def main(argv):

  # Set the parameters
  n_epochs = 5
  batch_size_train = 64
  batch_size_test = 1000
  learning_rate = 0.01
  momentum = 0.5
  log_interval = 10

  random_seed = 1
  torch.backends.cudnn.enabled = False
  torch.manual_seed(random_seed)

  # Load the training and test data
  train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('/Users/pranavraj/Desktop/visualstudio-opencv/files/', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
    batch_size=batch_size_train, shuffle=True)
  # Load the test data
  test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('/Users/pranavraj/Desktop/visualstudio-opencv/files/', train=True, download=True,                             transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
    batch_size=batch_size_test, shuffle=True)

  # Get the first batch of test data
  examples = enumerate(test_loader)
  batch_idx, (example_data, example_targets) = next(examples)

  example_data.shape

  # Plot the first 6 examples
  fig = plt.figure()
  for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
  plt.show()  # Add this line
  network = Net()
  optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                        momentum=momentum)
  # Train the network
  train_losses = []
  train_counter = []
  test_losses = []
  test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

  test(network, test_loader, test_losses)
  for epoch in range(1, n_epochs + 1):
    train(epoch, network, optimizer, train_loader, train_losses, train_counter, log_interval)
    test(network, test_loader, test_losses)

  # Plot the training and test losses
  fig = plt.figure()
  plt.plot(train_counter, train_losses, color='blue')
  plt.scatter(test_counter, test_losses, color='red')
  plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
  plt.xlabel('number of training examples seen')
  plt.ylabel('negative log likelihood loss')
  plt.show()


  with torch.no_grad():
    output = network(example_data)

  # Plot the first 6 examples with the prediction for each example
  fig = plt.figure()
  for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(
      output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
  plt.show()


  # Visualize the network
  x = torch.randn(1, 1, 28, 28)  
  y = network(x)
  make_dot(y, params=dict(network.named_parameters())).render("network", format="png")

if __name__ == "__main__":
    main(sys.argv)